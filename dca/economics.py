import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_cashflow(
    t_days: np.ndarray,
    rates: np.ndarray,
    oil_price: float,
    opex_fixed: float,
    opex_var: float,
    discount_rate_annual: float = 0.10,
    capex: float = 0.0
) -> pd.DataFrame:
    """
    Calculates monthly cash flow.
    t_days: assumed to be daily or similar time basis.
    We'll integrate rate over time to get volumes for periods.
    For simplicity, we calculate Daily Cash Flow and then aggregate?
    Or just Daily CF -> NPV.
    """
    
    # 1. Volume Calculation (Daily)
    # rate is bbl/d. 
    daily_revenue = rates * oil_price
    daily_opex = opex_fixed + (rates * opex_var)
    daily_cf = daily_revenue - daily_opex
    
    # Check if we should apply CAPEX at t=0
    # For now, we assume this is "Point Forward" economics (from today).
    # If t=0 matches today.
    
    # 2. Discount Factor
    # r_daily = (1 + r_annual)^(1/365) - 1
    # or continuous: exp(-r * t/365)
    r_daily = (1 + discount_rate_annual)**(1/365.0) - 1
    
    # discount factor = 1 / (1+r)^t
    df = 1.0 / ((1 + r_daily) ** t_days)
    
    discounted_cf = daily_cf * df
    
    # Apply CAPEX at day 0 (first point) or separate?
    # Let's subtract CAPEX from the sum of Discounted CF to get NPV
    npv = np.sum(discounted_cf) - capex
    
    # Cumulative Cash Flow (Undiscounted)
    cum_cf = np.cumsum(daily_cf)
    cum_cf[0] -= capex
    
    # Payout Time: First time cum_cf > 0
    payout_idx = np.argmax(cum_cf > 0)
    payout_days = t_days[payout_idx] if cum_cf[payout_idx] > 0 else None
    
    # ROI = (Total Undiscounted CF - CAPEX) / CAPEX
    # Total Revenue?
    mask_pos = daily_cf > 0 # Project life? No, run till end of forecast
    
    total_undisc_cf = np.sum(daily_cf) - capex
    roi = (total_undisc_cf / capex) if capex > 0 else 0.0
    
    return {
        "metrics": {
            "npv": npv,
            "roi": roi,
            "payout_days": payout_days
        },
        "series": {
            "t_days": t_days,
            "daily_cf": daily_cf,
            "cum_cf": cum_cf,
            "discounted_cf": discounted_cf
        }
    }
