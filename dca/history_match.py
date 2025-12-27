import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from .pvt import simulate_pvt_mbal

def auto_history_match(
    t_obs: np.ndarray,
    p_obs: np.ndarray, # Observed reservoir pressure if available, or just rate match
    rate_obs: np.ndarray
) -> Dict[str, Any]:
    """
    Finds best N (Oil In Place) and J (Productivity Index) to match observed Rates.
    Assumes simple liquid expansion model.
    """
    
    def objective(params):
        n_guess, j_guess = params
        if n_guess < 1000 or j_guess < 0.1: return 1e9
        
        # Simulate
        sim = simulate_pvt_mbal(t_obs, len(t_obs), pi=4000.0, n_oil_inplace=n_guess, compressibility=1e-5)
        # Note: simulation is rigid in structure, we adjust params to scale it
        # Actually our sim function uses hardcoded J and Pi.
        # We need to make simulate_pvt_mbal accept J. 
        # (Assuming we updated pvt.py or we shadow it here, let's just use simplified logic here for speed)
        
        # Re-implement simple logic for speed
        pi = 4000.0
        ce = 1e-5
        pwf = 1000.0
        dt = 1.0
        
        q_sim = []
        curr_p = pi
        cum = 0
        denom = n_guess * ce
        
        for _ in range(len(t_obs)):
            q = j_guess * (curr_p - pwf)
            if q < 0: q = 0
            q_sim.append(q)
            cum += q * dt
            curr_p = pi - (cum / denom)
            
        q_sim = np.array(q_sim)
        return np.mean((q_sim - rate_obs)**2) # MSE
        
    res = minimize(objective, x0=[1e6, 5.0], bounds=[(1e4, 1e8), (0.1, 100)], method='Nelder-Mead')
    
    return {
        "success": res.success,
        "best_n": res.x[0],
        "best_j": res.x[1],
        "final_mse": res.fun
    }
