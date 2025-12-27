import pytest
import numpy as np
import pandas as pd
from dca.probabilistic import generate_probabilistic_forecast
from dca.economics import calculate_cashflow
from dca.fit import clean_data_advanced

def test_probabilistic_forecast():
    # Test that P10 > P50 > P90 at some point?
    # Wait, in decline curve:
    # P10 is High Case (Optimistic) -> Higher Rates
    # P90 is Low Case (Conservative) -> Lower Rates
    # So q_P10 > q_P90
    
    t = np.arange(100)
    # Exp decline params: qi=1000, d=0.01
    params = [1000, 0.01]
    
    res = generate_probabilistic_forecast("Exponential", params, t, n_iter=100)
    
    assert "p10" in res
    assert "p90" in res
    
    # Check High Case > Low Case
    assert np.mean(res['p10']) > np.mean(res['p90'])
    
def test_economics_npv():
    # Simple case: 1 day, rate=100, price=50, opex=0, disc=0
    # CF = 100*50 = 5000. NPV = 5000 - CAPEX
    t = np.array([1.0]) # Day 1
    q = np.array([100.0])
    
    res = calculate_cashflow(t, q, oil_price=50.0, opex_fixed=0, opex_var=0, discount_rate_annual=0.0, capex=1000.0)
    
    # Discount factor for day 1 approx 1.0 (since rate 0)
    # Daily CF = 5000.
    # NPV = 5000 - 1000 = 4000
    
    # Float precision
    assert res["metrics"]["npv"] == pytest.approx(4000.0, rel=1e-3)
    
def test_cleaning_logic():
    # Create data with obvious outlier
    t = np.arange(20)
    q = np.full(20, 100.0)
    q[10] = 5000.0 # Outlier
    
    # Should detect 1 outlier?
    # IsolationForest needs enough samples. 20 might be small but let's try.
    # We put a safeguard < 50 samples in code?
    # Yes: if len(q) < 50: return ...
    
    t_clean, q_clean, mask = clean_data_advanced(t, q)
    if len(q) < 50:
         # Expect no cleaning
         assert len(q_clean) == 20
    else:
         pass
         
    # Test with larger data
    t2 = np.arange(100)
    q2 = np.full(100, 100.0)
    q2[50] = 10000.0
    
    t_cl, q_cl, mask = clean_data_advanced(t2, q2)
    # Should remove at least 1
    assert len(q_cl) < 100
    assert q2[50] not in q_cl
