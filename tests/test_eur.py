import pytest
import numpy as np
from dca.eur import calculate_eur, solve_t_econ, exponential_decline

def test_t_econ_solver():
    # Exp: q = 1000 * exp(-0.01 * t)
    # q_econ = 100
    # 100 = 1000 * exp(-0.01t) => 0.1 = exp(-0.01t) => ln(0.1) = -0.01t
    # t = -ln(0.1)/0.01 = -(-2.302)/0.01 = 230.25
    
    params = [1000, 0.01]
    t_econ = solve_t_econ(exponential_decline, params, q_econ=100.0, t_max=10000)
    assert t_econ == pytest.approx(230.258, rel=1e-3)

def test_t_econ_not_reached():
    # q_econ very low
    params = [1000, 0.01]
    # t(100) ~ 230 days. t_max = 100
    t_econ = solve_t_econ(exponential_decline, params, q_econ=100.0, t_max=100)
    assert t_econ == 100.0

def test_eur_exponential_analytic():
    # EUR = qi/d * (1 - exp(-dt))
    # Infinite EUR = 1000/0.01 = 100,000
    params = [1000, 0.01]
    res = calculate_eur("Exponential", params, q_econ=0.0001, t_max=50000) # Effectively infinite
    
    assert res["eur"] == pytest.approx(100000, rel=1e-2)

def test_eur_integration_cleanup():
    # Test logic when model name is unknown
    res = calculate_eur("Unknown", [], 10)
    assert "error" in res
