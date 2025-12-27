import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from typing import Dict, Any, Callable
from .models import exponential_decline, harmonic_decline, hyperbolic_decline

def solve_t_econ(model_func: Callable, params: list, q_econ: float, t_max: float) -> float:
    """
    Find t where q(t) = q_econ.
    If q(0) < q_econ, returns 0.
    If q(t_max) > q_econ, returns t_max.
    """
    # Check start
    q_start = model_func(0, *params)
    if q_start < q_econ:
        return 0.0
        
    # Check end
    q_end = model_func(t_max, *params)
    if q_end > q_econ:
        return t_max
        
    # Root finding
    try:
        def target(t):
            return model_func(t, *params) - q_econ
        t_root = brentq(target, 0, t_max)
        return float(t_root)
    except Exception:
        # Fallback if brentq fails (e.g. erratic function, shouldn't happen with smooth DCA)
        return t_max

def calculate_eur_exponential(qi: float, d: float, t_econ: float) -> float:
    # Integral q(t) dt from 0 to t_econ
    # = (qi/-D) * [exp(-Dt)]_0^t_econ = (qi/D) * (1 - exp(-D*t_econ))
    if abs(d) < 1e-9: # Limit as D->0 is linear (qi*t), but D is constrained > 0
        return qi * t_econ 
    return (qi / d) * (1.0 - np.exp(-d * t_econ))

def calculate_eur_harmonic(qi: float, d: float, t_econ: float) -> float:
    # Integral q(t) dt = (qi/D) * ln(1 + D*t)
    if abs(d) < 1e-9:
        return qi * t_econ
    return (qi / d) * np.log(1.0 + d * t_econ)

def calculate_eur_hyperbolic(qi: float, d: float, b: float, t_econ: float) -> float:
    # Integral
    if abs(b) < 1e-4:
        return calculate_eur_exponential(qi, d, t_econ)
    if abs(b - 1.0) < 1e-4:
        return calculate_eur_harmonic(qi, d, t_econ)
        
    # General Case b != 1
    # Int (1+bDt)^(-1/b) dt
    # let u = 1+bDt, du = bD dt => dt = du/bD
    # = (qi/bD) * Int u^(-1/b) du
    # = (qi/bD) * [ u^(-1/b + 1) / (-1/b + 1) ]
    # Exponent = (b-1)/b
    # Factor = 1 / ((b-1)/b) = b/(b-1)
    # Total prefactor = (qi/bD) * (b/(b-1)) = qi / (D*(b-1))
    # Result = (qi / (D*(b-1))) * [ (1+bDt)^((b-1)/b) - 1 ]
    # Or typically written as: (qi^b) / ((1-b)D) * [ q_limit^(1-b) - qi^(1-b) ] ... various forms.
    # Using the time form:
    # = (qi / (D*(1-b))) * [ 1 - (1 + b*D*t_econ)^((b-1)/b) ]
    
    term = (1.0 + b * d * t_econ) ** ((b - 1.0) / b)
    numerator = qi * (1.0 - term)
    denominator = d * (1.0 - b)
    return numerator / denominator

def calculate_eur(model_name: str, params: list, q_econ: float, t_max: float = 3650.0) -> Dict[str, Any]:
    """
    Calculate t_econ and EUR.
    """
    t_econ = 0.0
    eur_val = 0.0
    reached = True
    
    # 1. Determine Model Function
    if model_name == "Exponential":
        model_func = exponential_decline
    elif model_name == "Harmonic":
        model_func = harmonic_decline
    elif model_name == "Hyperbolic":
        model_func = hyperbolic_decline
    else:
        return {"error": "Unknown model"}

    # 2. Calculate t_econ
    # For actual EUR integration, we use the smaller of t_econ (physically econ limit) or t_max (forecast horizon)
    # But usually EUR is defined to the economic limit.
    # The requirement says: "If q(t_max) > q_econ => t_econ=t_max and mark as 'not reached'"
    t_lim = solve_t_econ(model_func, params, q_econ, t_max)
    
    if t_lim >= t_max and model_func(t_max, *params) > q_econ:
        reached = False
        t_lim = t_max
        
    # 3. Calculate EUR
    # Use analytic formulas
    try:
        if model_name == "Exponential":
            eur_val = calculate_eur_exponential(params[0], params[1], t_lim)
        elif model_name == "Harmonic":
            eur_val = calculate_eur_harmonic(params[0], params[1], t_lim)
        elif model_name == "Hyperbolic":
            eur_val = calculate_eur_hyperbolic(params[0], params[1], params[2], t_lim)
    except Exception:
        # Fallback to numeric integration
        eur_val, _ = quad(model_func, 0, t_lim, args=tuple(params))
        
    return {
        "t_econ": t_lim,
        "eur": eur_val,
        "econ_limit_reached": reached
    }
