import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any

def analyze_rta(
    t_days: np.ndarray,
    q: np.ndarray,
    p: np.ndarray = None,
    pi: float = 3000.0,
    ct: float = 1e-5
) -> Dict[str, Any]:
    """
    Performs simplified Rate Transient Analysis (Blasingame / Agarwal-Gardner).
    If pressure 'p' is None, we synthesize it for demonstration.
    
    Calculates:
    - Material Balance Time (t_c) = Cum / Rate
    - Normalized Rate (q / delta_p)
    """
    
    # 1. Synthesize Pressure if missing
    if p is None:
        # Assume generic pressure decline
        # P(t) = Pi * exp(-0.001*t)
        p = pi * np.exp(-0.0005 * t_days)
    
    delta_p = pi - p
    delta_p[delta_p < 1] = 1.0 # Avoid div by zero
    
    # 2. Material Balance Time (MBT)
    cum_q = np.cumsum(q)  # Approximation without integration time steps for simplicity
    # Better: cumtrapz
    from scipy.integrate import cumulative_trapezoid
    if len(t_days) > 1:
        cum_q = cumulative_trapezoid(q, t_days, initial=0)
        # Fix first point
        cum_q[0] = q[0] # Small approx
    
    q_safe = q.copy()
    q_safe[q_safe < 1e-5] = 1e-5
    
    t_mb = cum_q / q_safe
    
    # 3. Normalized Rate (q / delta_p)
    q_norm = q / delta_p
    
    # 4. Normalized Pressure (delta_p / q) - Inverse
    p_norm = delta_p / q_safe
    
    return {
        "t_mb": t_mb,
        "q_norm": q_norm,
        "p_norm": p_norm,
        "pressure": p,
        "delta_p": delta_p
    }

def create_rta_plot(results: Dict[str, Any]) -> go.Figure:
    """
    Creates a Log-Log RTA Type Curve Plot.
    """
    t_mb = results["t_mb"]
    q_norm = results["q_norm"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_mb, 
        y=q_norm, 
        mode='markers', 
        name='Data (q/Δp)',
        marker=dict(size=5, color='purple')
    ))
    
    # Synthetic "Type Curve" for show (Unit Slope line)
    # y = 1/x for Flow Regime identification
    # Just draw a referencing line
    x_ref = np.logspace(np.log10(max(1, t_mb.min())), np.log10(t_mb.max()), 100)
    y_ref = q_norm.max() * (x_ref[0] / x_ref) # Unit slope approx (Boundary Dominated Flow)
    
    fig.add_trace(go.Scatter(
        x=x_ref,
        y=y_ref,
        mode='lines',
        name='Boundary Dominated (Slope -1)',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_xaxes(type="log", title="Material Balance Time (days)")
    fig.update_yaxes(type="log", title="Normalized Rate (q / Δp)")
    
    fig.update_layout(
        title="RTA: Blasingame Type Curve Plot",
        template="plotly_white"
    )
    
    return fig
