import numpy as np
from scipy.optimize import curve_fit
from .models import exponential_decline
from typing import Dict, Any, Tuple

def fit_simple_exponential(t: np.ndarray, q: np.ndarray) -> float:
    """
    Fits exponential and returns D. 
    Robust against small datasets or flat lines.
    """
    if len(t) < 3:
        return 0.0
        
    try:
        qi_guess = np.max(q)
        # Force D positive
        p0 = [qi_guess, 0.001]
        bounds = ([1e-5, 0.0], [np.inf, np.inf])
        popt, _ = curve_fit(exponential_decline, t, q, p0=p0, bounds=bounds, maxfev=2000)
        return popt[1]
    except Exception:
        return 0.0

def check_anomaly(t: np.ndarray, q: np.ndarray, short_window: int = 60, long_window: int = 240) -> Dict[str, Any]:
    """
    Check for anomaly by comparing decline rate in short tail vs long tail.
    Rule: if D_short / D_long >= 2.0 => Warning.
    """
    if len(t) == 0:
        return {"status": "unknown", "ratio": 0.0, "reason": "No data"}
        
    t_end = t[-1]
    
    # Select windows
    mask_short = t >= (t_end - short_window)
    mask_long = t >= (t_end - long_window)
    
    if np.sum(mask_short) < 5 or np.sum(mask_long) < 10:
         return {"status": "unknown", "ratio": 0.0, "reason": "Insufficient data"}
         
    # Shift time to start at 0 for numeric stability in fitting (optional but good practice)
    # But for a simple D fit, shift doesn't affect D.
    
    d_short = fit_simple_exponential(t[mask_short] - t[mask_short].min(), q[mask_short])
    d_long = fit_simple_exponential(t[mask_long] - t[mask_long].min(), q[mask_long])
    
    if d_long < 1e-6:
        ratio = 0.0 # prevent div by zero
        # If d_long is basically 0 (flat), and d_short is high, it's definitely an anomaly
        if d_short > 0.01:
            ratio = 999.0 
    else:
        ratio = d_short / d_long
        
    status = "normal"
    reason = "Decline rate consistent"
    
    if ratio >= 2.0:
        status = "warning"
        reason = f"Short-term decline ({d_short:.4f}) is >= 2x long-term ({d_long:.4f})"
        
    return {
        "status": status,
        "ratio": ratio,
        "reason": reason,
        "d_short": d_short,
        "d_long": d_long
    }
