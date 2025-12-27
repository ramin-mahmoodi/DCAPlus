import pandas as pd
from datetime import datetime
from typing import Dict, Any

def generate_summary_report(
    qc_report: Dict[str, Any],
    fit_results: Dict[str, Any],
    eur_results: Dict[str, Any],
    anomaly_results: Dict[str, Any],
    q_econ: float
) -> Dict[str, Any]:
    """
    Consolidates analysis into a single dictionary for reporting/download.
    """
    summary = {
        "report_date": datetime.now().isoformat(),
        "input_qc": qc_report,
        "input_parameters": {
            "q_econ": q_econ,
        },
        "model_selection": {
            "best_model": fit_results.get("best_model_name", "None"),
            "params": fit_results.get("best_model_data", {}).get("params", []),
            "metrics": fit_results.get("best_model_data", {}).get("metrics", {})
        },
        "economics": {
            "eur_volume": eur_results.get("eur", 0.0),
            "t_econ_days": eur_results.get("t_econ", 0.0),
            "econ_limit_reached": eur_results.get("econ_limit_reached", False)
        },
        "anomaly_detection": {
             "status": anomaly_results.get("status", "unknown"),
             "ratio": anomaly_results.get("ratio", 0.0),
             "reason": anomaly_results.get("reason", "")
        }
    }
    return summary

def export_model_comparison(fit_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a DataFrame comparing all models (RMSE, MAE, AIC).
    """
    if not fit_results.get("success", False):
        return pd.DataFrame()
        
    rows = []
    all_res = fit_results.get("all_results", {})
    for model, data in all_res.items():
        if data["success"]:
            m = data["metrics"]
            rows.append({
                "Model": model,
                "RMSE": m["rmse"],
                "MAE": m["mae"],
                "AIC": m["aic"],
                "Params": str([round(p, 4) for p in data["params"]])
            })
        else:
            rows.append({
                "Model": model,
                "RMSE": None,
                "MAE": None,
                "AIC": None,
                "Params": "Fit Failed"
            })
            
            
    return pd.DataFrame(rows)

from fpdf import FPDF
import tempfile

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'DCA-Plus Analysis Report', border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(
    summary_data: Dict[str, Any],
    forecast_df: pd.DataFrame
) -> bytes:
    """
    Generates a PDF report as bytes.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font("helvetica", size=12)
    
    # 1. Summary Info
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.set_font("helvetica", size=11)
    
    econ = summary_data.get("economics", {})
    model = summary_data.get("model_selection", {})
    qc = summary_data.get("input_qc", {})
    
    pdf.cell(0, 8, f"Report Date: {summary_data.get('report_date', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Best Model: {model.get('best_model', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Estimated Reserves (EUR): {econ.get('eur_volume', 0):,.0f} bbl", ln=True)
    pdf.cell(0, 8, f"Remaining Life: {econ.get('t_econ_days', 0):,.0f} days", ln=True)
    pdf.ln(5)
    
    # 2. Data Quality
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Data Quality Control", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, f"Raw Data Points: {qc.get('n_raw', 0)}", ln=True)
    pdf.cell(0, 8, f"Used for Fit: {qc.get('n_fit', 0)}", ln=True)
    pdf.cell(0, 8, f"Outliers Flagged: {qc.get('outliers_flagged', 0)}", ln=True)
    pdf.ln(5)
    
    # 3. Model Parameters
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Model Parameters", ln=True)
    pdf.set_font("helvetica", size=11)
    params = model.get("params", [])
    pdf.cell(0, 8, f"Parameters: {[round(p, 5) for p in params]}", ln=True)
    metrics = model.get("metrics", {})
    pdf.cell(0, 8, f"RMSE: {metrics.get('rmse', 0):.4f}", ln=True)
    pdf.cell(0, 8, f"AIC: {metrics.get('aic', 0):.4f}", ln=True)
    pdf.ln(5)
    
    # 4. Forecast Preview
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "4. Forecast Preview (First 12 Months)", ln=True)
    pdf.set_font("courier", size=9)
    
    # Table Header
    pdf.cell(40, 7, "Date", border=1)
    pdf.cell(40, 7, "Rate (bbl/d)", border=1, ln=True)
    
    # Table Rows (First 12 rows ~ 1 year monthly if monthly, or first 12 days? forecast_df is daily)
    # Let's take every 30th day
    subset = forecast_df.iloc[::30].head(12)
    for _, row in subset.iterrows():
        d_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
        pdf.cell(40, 7, d_str, border=1)
        pdf.cell(40, 7, f"{row['oil_rate']:.2f}", border=1, ln=True)
        
    return bytes(pdf.output())
