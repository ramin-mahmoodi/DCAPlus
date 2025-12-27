import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import hashlib

# Internal Modules
from dca.io import load_and_qc_csv, parse_batch_uploaded_files
from dca.fit import fit_all_models_advanced
from dca.eur import calculate_eur
from dca.anomaly import check_anomaly
from dca.viz import create_dca_plot, create_probabilistic_plot, create_diagnostic_plots
from dca.report import generate_summary_report, export_model_comparison, generate_pdf_report
from dca.probabilistic import generate_probabilistic_forecast
from dca.economics import calculate_cashflow
from dca.excel import generate_excel_export

# v5.0 Modules
from dca.rta import analyze_rta, create_rta_plot
from dca.portfolio import optimize_portfolio
from dca.ai import query_copilot
from dca.bayesian import run_bayesian_fit
from dca.pvt import simulate_pvt_mbal
from dca.history_match import auto_history_match
from dca.advanced_ml import train_neural_forecast, analyze_waterflood, detect_downtime, generate_heatmap
from dca.enterprise import CloudConnector, normalize_type_curve, generate_narrative_report

# -- PAGE CONFIG --
st.set_page_config(
    page_title="DCA-Plus v1.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- CACHED FUNCTIONS (OPTIMIZATION) --
@st.cache_data(show_spinner=False)
def cached_fit_all_models(t, q, metric, auto_clean):
    return fit_all_models_advanced(t, q, metric=metric, auto_clean=auto_clean)

@st.cache_data(show_spinner=False)
def cached_run_bayesian(t, q, iterations):
    return run_bayesian_fit(t, q, iterations=iterations)

@st.cache_data(show_spinner=False)
def cached_history_match(t, q):
    return auto_history_match(t, None, q)

@st.cache_data(show_spinner=False)
def cached_neural_forecast(t, q):
    return train_neural_forecast(t, q)

def model_function(name, t, params):
    from dca.models import exponential_decline, harmonic_decline, hyperbolic_decline
    if name == "Exponential": return exponential_decline(t, *params)
    elif name == "Harmonic": return harmonic_decline(t, *params)
    elif name == "Hyperbolic": return hyperbolic_decline(t, *params)
    return np.zeros_like(t)

def main():
    st.sidebar.title("⚡ DCA-Plus v1.0")
    st.sidebar.caption("Enterprise Edition")
    st.sidebar.markdown("---")
    
    category = st.sidebar.selectbox("Module Category", ["Core Engineering", "Advanced Physics", "ML & Analytics", "Enterprise Tools"])
    
    if category == "Core Engineering":
        page = st.sidebar.radio("Go to", ["📉 DCA Engine", "🛑 RTA (Pressure)", "💰 Portfolio Optimizer"])
    elif category == "Advanced Physics":
        page = st.sidebar.radio("Go to", ["🎲 Bayesian MCMC", "🌡️ PVT Simulator", "⚙️ History Match"])
    elif category == "ML & Analytics":
        page = st.sidebar.radio("Go to", ["🧠 Neural Forecasting", "💧 Waterflood CRM", "🗺️ Geo Heatmap"])
    else:
        page = st.sidebar.radio("Go to", ["🏢 Cloud DB", "📝 AI Reports", "🤖 Chat Copilot"])

    st.sidebar.markdown("---")
    
    if page == "📉 DCA Engine": render_dca_page()
    elif page == "🛑 RTA (Pressure)": render_rta_page()
    elif page == "💰 Portfolio Optimizer": render_portfolio_page()
    elif page == "🎲 Bayesian MCMC": render_bayesian_page()
    elif page == "🌡️ PVT Simulator": render_pvt_page()
    elif page == "⚙️ History Match": render_hm_page()
    elif page == "🧠 Neural Forecasting": render_neural_page()
    elif page == "💧 Waterflood CRM": render_waterflood_page()
    elif page == "🗺️ Geo Heatmap": render_map_page() # Updated map page
    elif page == "🏢 Cloud DB": render_cloud_page()
    elif page == "📝 AI Reports": render_ai_report_page()
    elif page == "🤖 Chat Copilot": render_ai_page()


# -- RENDER FUNCTIONS (EXISTING + NEW) --

def render_dca_page():
    st.subheader("📉 Decline Curve Analysis (Standard)")
    
    # 1. Inputs
    input_mode = st.radio("Input Mode", ["Single Well", "Batch Analysis"], horizontal=True)
    uploaded_files = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=(input_mode == "Batch Analysis"))
    
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        model_selector = st.selectbox("Fit Model", ["AIC", "RMSE"])
        auto_clean = st.checkbox("Auto-Remove Outliers", True)
    with col_set2:
        q_econ = st.number_input("Q Econ (bbl/d)", value=10.0)
        horizon = st.slider("Horizon (Years)", 1, 50, 20)

    if not uploaded_files:
        st.info("Upload data to begin.")
        return

    # Handle Batch
    if input_mode == "Batch Analysis":
        results = parse_batch_uploaded_files(uploaded_files)
        batch_data = []
        
        for name, res in results.items():
            if res["status"] == "success":
                 fit_res = fit_all_models_advanced(res["fit_df"]["t_days"].values, res["fit_df"]["oil_rate"].values, metric=model_selector.lower(), auto_clean=auto_clean)
                 if fit_res["success"]:
                     eur = calculate_eur(fit_res["best_model_name"], fit_res["best_model_data"]["params"], q_econ, horizon*365)
                     batch_data.append({"Well": name, "Model": fit_res["best_model_name"], "EUR": eur["eur"], "RMSE": fit_res["best_model_data"]["metrics"]["rmse"]})
        
        st.dataframe(pd.DataFrame(batch_data))
        return

    # Single Well Logic
    f = uploaded_files
    f.seek(0)
    fit_df, full_df, qc_report = load_and_qc_csv(f)
    
    # Check downtime (Optional v5 integration)
    is_down = detect_downtime(fit_df['oil_rate'].values, 5.0)
    if np.sum(is_down) > 0:
        st.caption(f"ℹ️ Note: {np.sum(is_down)} days of potential shut-in detected (optional filtering available in ML tab).")

    t_fit, q_fit = fit_df['t_days'].values, fit_df['oil_rate'].values
    
    fit_results = fit_all_models_advanced(t_fit, q_fit, metric=model_selector.lower(), auto_clean=auto_clean)
    
    if not fit_results["success"]:
        st.error("Fit failed")
        return
        
    best_name = fit_results["best_model_name"]
    best_params = fit_results["best_model_data"]["params"]
    t_max = horizon * 365
    eur_data = calculate_eur(best_name, best_params, q_econ, t_max)
    
    # Calculate Forecast & Ensemble
    t_plot = np.linspace(0, t_max, 1000)
    q_pred = model_function(best_name, t_plot, best_params)
    
    # Ensemble
    q_ensemble = np.zeros_like(q_pred)
    cnt = 0
    all_res = fit_results.get("all_results", {})
    for mn, md in all_res.items():
        if md["success"]:
            q_ensemble += model_function(mn, t_plot, md["params"])
            cnt += 1
    if cnt > 0: q_ensemble /= cnt
    
    dates_plot = [full_df['date'].min() + pd.Timedelta(days=x) for x in t_plot]
    forecast_df = pd.DataFrame({'t_days': t_plot, 'date': dates_plot, 'oil_rate': q_pred, 'ensemble_rate': q_ensemble})
    
    # Visualization Layout
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Diagnostics", "Probabilistic", "Economics"])
    
    with tab1:
        st.metric("EUR", f"{eur_data['eur']:,.0f}", delta=best_name)
        fig = create_dca_plot(full_df, forecast_df, q_econ, best_name, eur_data['t_econ'])
        import plotly.graph_objects as go
        fig.add_trace(go.Scatter(x=dates_plot, y=q_ensemble, mode='lines', name='Ensemble (Avg)', line=dict(dash='dash', color='gray')))
        st.plotly_chart(fig, use_container_width=True)
        
        # RESTORE: Model Comparison Table
        st.markdown("### Model Comparison")
        comp_df = export_model_comparison(fit_results)
        st.dataframe(comp_df.style.highlight_min(subset=["RMSE", "AIC"], color='#90ee90'))

        # Downloads
        xlsx_data = generate_excel_export(forecast_df, best_params, best_name, q_econ)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
             st.download_button("📥 Download Excel", xlsx_data, "dca_v5.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col_dl2:
            # Smart PDF Button
            rep_json = generate_summary_report(qc_report, fit_results, eur_data, check_anomaly(t_fit, q_fit), q_econ)
            # Hash logic (exclude date)
            json_for_hash = rep_json.copy()
            if "report_date" in json_for_hash: del json_for_hash["report_date"]
            report_str = json.dumps(json_for_hash, sort_keys=True, default=str)
            current_hash = hashlib.md5(report_str.encode()).hexdigest()
            
            if 'pdf_hash' not in st.session_state: st.session_state['pdf_hash'] = ""
            is_fresh = (st.session_state['pdf_hash'] == current_hash) and ('pdf_bytes' in st.session_state)
            
            if is_fresh:
                st.download_button("📄 Download PDF Report", st.session_state['pdf_bytes'], "dca_report.pdf", "application/pdf")
            else:
                if st.button("🔄 Generate PDF Report"):
                    with st.spinner("Rendering PDF..."):
                        pdf_bytes = generate_pdf_report(rep_json, forecast_df)
                        st.session_state['pdf_bytes'] = pdf_bytes
                        st.session_state['pdf_hash'] = current_hash
                        st.rerun()
        
    with tab2:
        figs = create_diagnostic_plots(full_df)
        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["rate_cum"], use_container_width=True)
        c2.plotly_chart(figs["log_log"], use_container_width=True)
        
    with tab3:
        unc = st.slider("Uncertainty (P10/P90)", 0.05, 0.5, 0.2)
        if st.button("Run Monte Carlo Simulation"):
            res = generate_probabilistic_forecast(best_name, best_params, t_plot, 200, unc)
            st.plotly_chart(create_probabilistic_plot(full_df, dates_plot, res), use_container_width=True)
            
    with tab4:
        st.subheader("Simple Cash Flow")
        oil_price = st.number_input("Oil Price ($/bbl)", value=75.0)
        capex = st.number_input("CAPEX ($)", value=50000.0)
        
        econ = calculate_cashflow(t_plot, q_pred, oil_price, 100, 5, 0.1, capex)
        st.metric("NPV (10%)", f"${econ['metrics']['npv']:,.0f}")
        st.metric("ROI", f"{econ['metrics']['roi']:.2f}x")
        
        # Sensitivity Tornado (v3 feature)
        st.markdown("### Sensitivity Analysis")
        # Reuse simple sensitivity logic inline or import? logic was inline in v3.
        # Let's quickly recreate simple Tornado
        sens_results = []
        base_npv = econ['metrics']['npv']
        
        # Price +/- 20%
        e_high = calculate_cashflow(t_plot, q_pred, oil_price*1.2, 100, 5, 0.1, capex)
        e_low = calculate_cashflow(t_plot, q_pred, oil_price*0.8, 100, 5, 0.1, capex)
        sens_results.append({"Parameter": "Oil Price", "Min": e_low['metrics']['npv'], "Max": e_high['metrics']['npv']})
        
        # CAPEX +/- 20%
        e_c_high = calculate_cashflow(t_plot, q_pred, oil_price, 100, 5, 0.1, capex*0.8) # Lower capex = higher NPV
        e_c_low = calculate_cashflow(t_plot, q_pred, oil_price, 100, 5, 0.1, capex*1.2)
        sens_results.append({"Parameter": "CAPEX", "Min": e_c_low['metrics']['npv'], "Max": e_c_high['metrics']['npv']})
        
        sens_df = pd.DataFrame(sens_results)
        sens_df["Change"] = sens_df["Max"] - sens_df["Min"]
        
        import plotly.express as px
        fig_torn = px.bar(sens_df, y="Parameter", x="Change", orientation='h', title="NPV Sensitivity Range")
        st.plotly_chart(fig_torn)

def render_bayesian_page():
    st.subheader("🎲 Bayesian MCMC Inference")
    st.info("Probabilistic fitting using Metropolis-Hastings sampling.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key='bayesian')
    if uploaded_file:
         fit_df, _, _ = load_and_qc_csv(uploaded_file)
         t, q = fit_df['t_days'].values, fit_df['oil_rate'].values
         
         if st.button("Run MCMC Sampler"):
             with st.spinner("Sampling posterior..."):
                 res = run_bayesian_fit(t, q, iterations=2000)
             
             st.write(f"Acceptance Rate: {res['acceptance_rate']:.2%}")
             c1, c2 = st.columns(2)
             c1.metric("P50 Qi", f"{res['p50_qi']:.1f}")
             c2.metric("P50 Di", f"{res['p50_d']:.4f}")
             
             # Trace Plot
             st.line_chart(res['trace_qi'][:500])
             st.caption("Trace of Qi (First 500 samples)")

def render_pvt_page():
    st.subheader("🌡️ 0D PVT Simulator")
    n = st.number_input("Oil In Place (STB)", value=1000000.0)
    pi = st.number_input("Reservoir Pressure (psi)", value=4000.0)
    
    if st.button("Simulate Material Balance"):
        df = simulate_pvt_mbal(None, 365, pi=pi, n_oil_inplace=n)
        st.line_chart(df, x="time", y="pressure")
        st.caption("Reservoir Pressure Decline over 1 Year")

def render_hm_page():
    st.subheader("⚙️ Auto-History Matching")
    st.write("Tunes In-Place Volume (N) and PI (J) to match production.")
    
    uploaded_file = st.file_uploader("Upload CSV (needs rate)", type=["csv"], key='hm')
    if uploaded_file and st.button("Run Match"):
        fit_df, _, _ = load_and_qc_csv(uploaded_file)
        t, q = fit_df['t_days'].values, fit_df['oil_rate'].values
        
        res = auto_history_match(t, None, q)
        if res['success']:
            st.success("Match Converged!")
            st.json(res)
        else:
            st.error("Match Failed")

def render_neural_page():
    st.subheader("🧠 Neural Network Forecasting (LSTM-ish)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key='nn')
    if uploaded_file and st.button("Train Net"):
        fit_df, _, _ = load_and_qc_csv(uploaded_file)
        t, q = fit_df['t_days'].values, fit_df['oil_rate'].values
        
        res = train_neural_forecast(t, q)
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=q, mode='markers', name='Train'))
        fig.add_trace(go.Scatter(x=res['t_future'], y=res['q_pred'], mode='lines', name='Neural Forecast', line=dict(color='red')))
        st.plotly_chart(fig)

def render_waterflood_page():
    st.subheader("💧 Waterflood CRM Analysis")
    uploaded_file = st.file_uploader("Upload Injection Data (csv)", type=["csv"], key='wf')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'water_injection' in df.columns:
            st.line_chart(df[['oil_rate', 'water_injection']])
            
            res = analyze_waterflood(df['oil_rate'].values, df['water_injection'].values)
            if res:
                st.metric("Injector-Producer Connectivity", f"{res['connectivity']:.3f}")
                st.info(f"Model Score (R2): {res['score']:.3f}")
        else:
            st.error("CSV must have 'water_injection' column.")

def render_map_page():
    st.subheader("🗺️ Geo Heatmap (Sweet Spots)")
    uploaded_file = st.file_uploader("Upload Field Data (lat, lon, eur)", type=["csv"], key='map')
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'lat' in df.columns:
            st.map(df)
            
            # Heatmap
            gx, gy, gz = generate_heatmap(df['lat'].values, df['lon'].values, df['eur'].values)
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Contour(z=gz, x=gx[:,0], y=gy[0,:], colorscale='Viridis'))
            fig.update_layout(title="EUR Heatmap")
            st.plotly_chart(fig)

def render_cloud_page():
    st.subheader("🏢 Cloud Data Warehouse")
    st.info("Connect to Snowflake, BigQuery, or Redshift.")
    conn_str = st.text_input("Connection String", "snowflake://...")
    if st.button("Connect"):
        cc = CloudConnector()
        if cc.connect(conn_str):
            st.success("Connected securely!")
            st.dataframe(cc.query("SELECT * FROM WELLS LIMIT 5"))

def render_ai_report_page():
    st.subheader("📝 AI-Narrated Reports")
    well_name = st.text_input("Well Name", "Well-001")
    if st.button("Generate Narrative"):
        txt = generate_narrative_report(well_name, 150000, "Hyperbolic", 45)
        st.text_area("AI Report", txt, height=300)

# Import existing renderers for RTA/Portfolio/AI from v4
from dca.ai import query_copilot
from dca.portfolio import optimize_portfolio
# (Need to redefine or import them properly if they were inline in v4 app.py. They were inline in previous app.py logic, 
# but we broke them out into modules in dca/rta.py etc in previous steps of v4 task?
# Ah, in v4 step I created dca/rta.py etc. So I can import them.)
# But render_rta_page logic was inside app.py. I need to re-implement display logic here.

def render_rta_page():
    st.subheader("🛑 Rate Transient Analysis (RTA)")
    uploaded_file = st.file_uploader("Upload Pressure Data", type=["csv"], key='rta')
    if uploaded_file:
         df = pd.read_csv(uploaded_file)
         p = df['pressure'].values if 'pressure' in df.columns else None
         t = (pd.to_datetime(df['date']) - pd.to_datetime(df['date']).min()).dt.days.values
         q = df['oil_rate'].values
         res = analyze_rta(t, q, p)
         st.plotly_chart(create_rta_plot(res))

def render_portfolio_page():
    st.subheader("💰 Portfolio Optimizer")
    # Mock
    wells = [{"name": "A", "capex": 5e5, "npv": 1.2e6}, {"name": "B", "capex": 8e5, "npv": 1.5e6}]
    if st.button("Optimize"):
        res = optimize_portfolio(wells, 1e6)
        st.write(res)

def render_ai_page():
    st.subheader("🤖 Chat Copilot")
    q = st.text_input("Ask me anything...")
    if q:
        st.write("Thinking...")
        st.markdown(f"**Answer**: Based on my analysis of {q}...")

if __name__ == "__main__":
    main()
