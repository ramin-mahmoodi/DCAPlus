import numpy as np
from typing import Dict, Any, List
from .models import exponential_decline, harmonic_decline, hyperbolic_decline

def generate_probabilistic_forecast(
    model_name: str,
    params: List[float],
    t_forecast: np.ndarray,
    n_iter: int = 1000,
    uncertainty_pct: float = 0.20
) -> Dict[str, Any]:
    """
    Generates P10, P50, P90 forecasts using Monte Carlo simulation on parameters.
    Assumes parameters have independent normal distributions with sigma = value * uncertainty_pct.
    """
    
    # 1. Identify Model Function
    if model_name == "Exponential":
        func = exponential_decline
        # params: qi, d
    elif model_name == "Harmonic":
        func = harmonic_decline
        # params: qi, d
    elif model_name == "Hyperbolic":
        func = hyperbolic_decline
        # params: qi, d, b
    else:
        return {}

    # 2. Generate Parameter Realizations
    # We clip parameters to physical bounds (qi>0, d>0, 0<b<=2)
    
    realizations = np.zeros((n_iter, len(t_forecast)))
    
    for i in range(n_iter):
        p_iter = []
        for idx, p_val in enumerate(params):
            # Normal distribution around p_val
            # sigma = p_val * uncertainty_pct (e.g. 20% uncertainty)
            sigma = p_val * uncertainty_pct
            val = np.random.normal(p_val, sigma)
            
            # Constraints
            if val <= 0: val = 1e-5 # qi, d must be positive
            
            # Specific constraint for 'b' in Hyperbolic (index 2)
            if model_name == "Hyperbolic" and idx == 2:
                val = np.clip(val, 0.01, 2.0)
                
            p_iter.append(val)
            
        realizations[i, :] = func(t_forecast, *p_iter)
        
    # 3. Calculate Percentiles
    p10 = np.percentile(realizations, 10, axis=0) # Pessimistic (Low case) - Wait, P10 usually means 90% probability of exceeding?
    # In O&G:
    # P90: 90% probability of being greater than this volume (Conservative/Low Estimate)
    # P10: 10% probability of being greater than this volume (Optimistic/High Estimate)
    # So np.percentile(10) is the value where 10% of data is below it -> This is P90 Estimate (Low Case)
    # np.percentile(90) -> This is P10 Estimate (High Case)
    
    # Let's return explicit curves
    q_p90 = np.percentile(realizations, 10, axis=0)
    q_p50 = np.percentile(realizations, 50, axis=0)
    q_p10 = np.percentile(realizations, 90, axis=0)
    
    return {
        "p90": q_p90,
        "p50": q_p50,
        "p10": q_p10,
        "realizations_subset": realizations[:50, :].tolist() # Return first 50 for spaghetti plot if needed
    }
