import pytest
import numpy as np
from dca.fit import fit_all_models
from dca.models import hyperbolic_decline

def test_fit_synthetic_hyperbolic():
    # Generate synthetic data
    qi = 500
    d = 0.005
    b = 0.6
    t = np.arange(0, 365, 10)
    
    q_true = hyperbolic_decline(t, qi, d, b)
    # Add very small noise so fit is exact enough
    np.random.seed(42)
    q_noisy = q_true + np.random.normal(0, 1.0, size=len(t))
    
    results = fit_all_models(t, q_noisy, metric="rmse")
    
    assert results["success"] is True
    # Hyperbolic should likely be best or very close to it.
    # Note: With noise, sometimes 3-param hyperbolic is harder to fit than 2-param if b is masked,
    # but with low noise it should work.
    
    # Check if Hyberbolic fit itself succeeded
    hyp_res = results["all_results"]["Hyperbolic"]
    assert hyp_res["success"] is True
    
    ft_params = hyp_res["params"]
    # Check qi
    assert ft_params[0] == pytest.approx(qi, rel=0.15)
    # Check d
    assert ft_params[1] == pytest.approx(d, rel=0.3)
    # Check b
    assert ft_params[2] == pytest.approx(b, rel=0.3)

def test_fit_robustness():
    # Test with empty data or NaNs (though io.py should filter, fit logic handles it?)
    t = np.array([])
    q = np.array([])
    res = fit_all_models(t, q)
    assert res["success"] is False # or handles gracefully
