import numpy as np

def exponential_decline(t: np.ndarray, qi: float, d: float) -> np.ndarray:
    """
    Exponential decline model: q(t) = qi * exp(-d * t)
    
    Args:
        t: Time array (days)
        qi: Initial Rate (>0)
        d: Nominal decline rate (1/day) (>0)
        
    Returns:
        Production rate array
    """
    return qi * np.exp(-d * t)

def harmonic_decline(t: np.ndarray, qi: float, d: float) -> np.ndarray:
    """
    Harmonic decline model: q(t) = qi / (1 + d * t)
    
    Args:
        t: Time array (days)
        qi: Initial Rate (>0)
        d: Nominal decline rate (1/day) (>0)
        
    Returns:
        Production rate array
    """
    # Prevent division by zero if t is negative (should not happen in valid range)
    denom = 1 + d * t
    # Safety for very large d*t
    return qi / np.maximum(denom, 1e-9)

def hyperbolic_decline(t: np.ndarray, qi: float, d: float, b: float) -> np.ndarray:
    """
    Hyperbolic decline model: q(t) = qi / (1 + b * d * t)^(1/b)
    
    Args:
        t: Time array (days)
        qi: Initial Rate (>0)
        d: Nominal decline rate (1/day) (>0)
        b: Hyperbolic exponent (0 < b <= 2)
        
    Returns:
        Production rate array
    """
    # Safety: if b is essentially zero, treat as exponential to avoid divide by zero
    if abs(b) < 1e-4:
        return exponential_decline(t, qi, d)
        
    base = 1 + b * d * t
    return qi / (np.maximum(base, 1e-9) ** (1.0 / b))
