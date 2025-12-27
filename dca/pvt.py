import numpy as np
import pandas as pd
from typing import Dict, Any

def simulate_pvt_mbal(
    t_days: np.ndarray,
    n_days: int,
    pi: float = 4000.0,
    n_oil_inplace: float = 1e6, # STB
    compressibility: float = 1e-5
) -> pd.DataFrame:
    """
    0D Material Balance Simulation (Liquid Expansion).
    P = Pi * (1 - Np / (N * ce * deltaP_approx?)) -> Linear approximation
    Actually: Np = N * ce * (Pi - P)
    So: P(t) = Pi - (Np(t) / (N * ce))
    
    We simulate a well flowing at constant PI (Productivity Index).
    q = J * (P - Pwf)
    """
    
    pressure_hist = []
    rate_hist = []
    cum_hist = []
    
    current_p = pi
    cum_np = 0.0
    j_index = 5.0 # STB/d/psi
    pwf = 1000.0 # Constant flowing pressure
    
    dt = 1.0 # day
    
    for i in range(n_days):
        # Inflow
        q = j_index * (current_p - pwf)
        if q < 0: q = 0
        if current_p < pwf: q = 0
        
        # Material Balance Update
        # delta_Np = q * dt
        cum_np += q * dt
        
        # P = Pi - (Np / (N * ce))
        # N * ce = Pore Volume compressibility term
        # Let's say max liquid expansion volume is 10% of N
        # N * ce * Pi ~ volume
        
        denom = n_oil_inplace * compressibility
        drop = cum_np / denom
        current_p = pi - drop
        if current_p < 0: current_p = 0
        
        pressure_hist.append(current_p)
        rate_hist.append(q)
        cum_hist.append(cum_np)
        
    return pd.DataFrame({
        "time": np.arange(n_days),
        "pressure": pressure_hist,
        "rate": rate_hist,
        "cum_oil": cum_hist
    })
