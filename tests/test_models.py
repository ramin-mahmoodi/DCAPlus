import pytest
import numpy as np
from dca.models import exponential_decline, harmonic_decline, hyperbolic_decline

def test_exponential_decline():
    t = np.array([0, 100, 200])
    qi = 1000
    d = 0.01
    
    q = exponential_decline(t, qi, d)
    
    assert q[0] == pytest.approx(1000)
    assert q[1] == pytest.approx(1000 * np.exp(-1)) # 100*0.01 = 1
    assert np.all(q > 0)
    assert np.all(np.diff(q) < 0) # strictly decreasing

def test_harmonic_decline():
    t = np.array([0, 100])
    qi = 1000
    d = 0.01
    
    q = harmonic_decline(t, qi, d)
    assert q[0] == 1000
    assert q[1] == pytest.approx(1000 / 2.0) # 1 + 0.01*100 = 2

def test_hyperbolic_decline():
    t = np.array([0, 100])
    qi = 1000
    d = 0.01
    b = 0.5
    
    q = hyperbolic_decline(t, qi, d, b)
    assert q[0] == 1000
    # q = qi / (1 + bdt)^(1/b) = 1000 / (1 + 0.5)^2 = 1000 / 1.5^2 = 1000/2.25 = 444.44
    expected = 1000 / (1.5**2)
    assert q[1] == pytest.approx(expected)

def test_hyperbolic_limits():
    # b near 0 should behave like exponential
    t = np.array([100])
    qi = 1000
    d = 0.01
    
    q_hyp = hyperbolic_decline(t, qi, d, b=1e-5)
    q_exp = exponential_decline(t, qi, d)
    
    assert q_hyp[0] == pytest.approx(q_exp[0], rel=1e-3)
