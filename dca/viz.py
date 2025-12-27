import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any

def create_dca_plot(
    data_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    q_econ: float,
    model_name: str,
    t_econ: float
) -> go.Figure:
    """
    Creates the main plot with:
    1. Observed data (scatter)
    2. Fit + Forecast (line)
    3. Economic Limit (hline)
    4. EUR Area (filled)
    """
    fig = go.Figure()
    
    # 1. Observed Data
    # Use only clean points for clarity, or flag outliers?
    # Requirement: Clean architecture. Let's plot all valid q>0 points from data_df
    # In io.py we kept full_df_processed. Let's assume data_df here is the one used for plotting.
    
    # Plot Normal points
    mask_normal = ~data_df.get('is_outlier', pd.Series([False]*len(data_df)))
    fig.add_trace(go.Scatter(
        x=data_df.loc[mask_normal, 'date'],
        y=data_df.loc[mask_normal, 'oil_rate'],
        mode='markers',
        name='Observed',
        marker=dict(color='black', size=5, opacity=0.6)
    ))
    
    # Optional: Plot outliers distinctively
    if not mask_normal.all():
        fig.add_trace(go.Scatter(
            x=data_df.loc[~mask_normal, 'date'],
            y=data_df.loc[~mask_normal, 'oil_rate'],
            mode='markers',
            name='Outliers (Ignored)',
            marker=dict(color='red', symbol='x', size=6)
        ))

    # 2. Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['oil_rate'],
        mode='lines',
        name=f'Forecast ({model_name})',
        line=dict(color='blue', width=2)
    ))

    # 3. Economic Limit Line
    fig.add_hline(
        y=q_econ,
        line_dash="dash",
        line_color="green",
        annotation_text=f"q_econ = {q_econ}",
        annotation_position="bottom right"
    )

    # 4. EUR Shaded Area
    # Filter forecast to < t_econ
    # t_days in forecast needs to be available or we map back.
    # forecast_df should have 't_days' and 'date'
    if 't_days' in forecast_df.columns:
        mask_eur = forecast_df['t_days'] <= t_econ
        if mask_eur.any():
            fig.add_trace(go.Scatter(
                x=forecast_df.loc[mask_eur, 'date'],
                y=forecast_df.loc[mask_eur, 'oil_rate'],
                fill='tozeroy',
                mode='none',
                name='EUR Area',
                fillcolor='rgba(0, 255, 0, 0.1)',
                showlegend=True
            ))

    fig.update_layout(
        title="Production History & Forecast",
        xaxis_title="Date",
        yaxis_title="Oil Rate (bbl/d)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def create_probabilistic_plot(
    data_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    prob_results: Dict[str, Any]
) -> go.Figure:
    """
    Plots P10, P50, P90 confidence intervals.
    """
    fig = go.Figure()
    
    # Observed
    fig.add_trace(go.Scatter(
        x=data_df['date'],
        y=data_df['oil_rate'],
        mode='markers',
        name='Observed',
        marker=dict(color='black', size=4, opacity=0.5)
    ))
    
    # P90 (Lower)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=prob_results['p90'],
        mode='lines',
        name='P90 (Low)',
        line=dict(width=0),
        showlegend=False
    ))
    
    # P10 (Upper) - Fill to P90
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=prob_results['p10'],
        mode='lines',
        name='P10 - P90 Range',
        fill='tonexty',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(width=0),
    ))
    
    # P50 (Mid)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=prob_results['p50'],
        mode='lines',
        name='P50 (Best)',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title="Probabilistic Forecast (Monte Carlo)",
        xaxis_title="Date",
        yaxis_title="Rate (bbl/d)",
        template="plotly_white"
    )
    return fig

def create_diagnostic_plots(data_df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Creates Rate vs Cumulative and Rate vs Time (Log-Log) plots.
    """
    # Rate vs Cumulative
    # Calculate cumulative production
    # Assuming daily data, cum = cumsum(rate)
    df = data_df.copy()
    df = df.sort_values('date')
    df['cum_oil'] = df['oil_rate'].cumsum()
    
    # 1. Rate vs Cumulative
    # Linear-Log or Log-Log? Typically Rate vs Cum is analyzed on Cartesian or Semi-Log.
    # We'll do Rate (y) vs Cum (x)
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(x=df['cum_oil'], y=df['oil_rate'], mode='markers', name='Observed'))
    fig_rc.update_layout(title="Rate vs. Cumulative Production", xaxis_title="Cumulative Oil (bbl)", yaxis_title="Rate (bbl/d)")
    
    # 2. Log-Log Rate vs Time
    # Shift time to start at 1 to avoid log(0)
    t0 = df['date'].min()
    df['t_days'] = (df['date'] - t0).dt.days + 1
    
    fig_ll = go.Figure()
    fig_ll.add_trace(go.Scatter(x=df['t_days'], y=df['oil_rate'], mode='markers', name='Observed'))
    fig_ll.update_xaxes(type="log")
    fig_ll.update_yaxes(type="log")
    fig_ll.update_layout(title="Log-Log Rate vs. Time", xaxis_title="Time (Days)", yaxis_title="Rate (bbl/d)")
    
    return {
        "rate_cum": fig_rc,
        "log_log": fig_ll
    }
