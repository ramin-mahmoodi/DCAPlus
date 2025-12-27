import numpy as np
from typing import Dict, Any

def run_bayesian_fit(
    t: np.ndarray,
    q: np.ndarray,
    iterations: int = 1000
) -> Dict[str, Any]:
    """
    Metropolis-Hastings MCMC implementation for Exponential Decline.
    Model: q = qi * exp(-d * t)
    Unknowns: qi, d
    Likelihood: Gaussian (Normal errors)
    Priors: Uniform
    """
    
    # Init params
    current_qi = q[0] if len(q) > 0 else 1000.0
    current_d = 0.01
    current_sigma = 50.0 # Error std dev
    
    trace_qi = []
    trace_d = []
    accepted = 0
    
    def log_likelihood(qi, d, sigma, t, q_obs):
        if qi < 0 or d < 0 or sigma <= 0: return -np.inf
        q_pred = qi * np.exp(-d * t)
        rss = np.sum((q_obs - q_pred)**2)
        n = len(t)
        # Log likelihood for normal dist
        ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - (1/(2*sigma**2)) * rss
        return ll

    current_ll = log_likelihood(current_qi, current_d, current_sigma, t, q)
    
    for i in range(iterations):
        # Propose new step
        prop_qi = current_qi + np.random.normal(0, 50)
        prop_d = current_d + np.random.normal(0, 0.001)
        
        prop_ll = log_likelihood(prop_qi, prop_d, current_sigma, t, q)
        
        # Acceptance ratio
        if prop_ll > current_ll:
            accept = True
        else:
            p_accept = np.exp(prop_ll - current_ll)
            accept = np.random.rand() < p_accept
            
        if accept:
            current_qi = prop_qi
            current_d = prop_d
            current_ll = prop_ll
            accepted += 1
            
        trace_qi.append(current_qi)
        trace_d.append(current_d)
        
    return {
        "trace_qi": trace_qi,
        "trace_d": trace_d,
        "acceptance_rate": accepted / iterations,
        "p10_qi": np.percentile(trace_qi, 90),
        "p50_qi": np.percentile(trace_qi, 50),
        "p90_qi": np.percentile(trace_qi, 10),
        "p10_d": np.percentile(trace_d, 90),
        "p50_d": np.percentile(trace_d, 50),
        "p90_d": np.percentile(trace_d, 10),
    }
