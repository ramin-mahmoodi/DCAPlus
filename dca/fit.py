import numpy as np
import logging
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional, Any
from .models import exponential_decline, harmonic_decline, hyperbolic_decline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, and AIC.
    """
    n = len(y_true)
    if n == 0:
        return {"rmse": np.inf, "mae": np.inf, "aic": np.inf}
        
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    
    # AIC = n * ln(RSS/n) + 2k
    # Protect log(0)
    rss_safe = max(rss, 1e-9)
    aic = n * np.log(rss_safe / n) + 2 * n_params
    
    return {"rmse": rmse, "mae": mae, "aic": aic}

def fit_exponential(t: np.ndarray, q: np.ndarray) -> Dict[str, Any]:
    """Fit Exponential Decline."""
    # Bounds: qi > 0, d > 0
    # Initial Guess: qi = max(q), d = simple linear slope est or default
    try:
        qi_guess = np.max(q)
        # Approximate d from first and last point (very rough)
        if len(t) > 1 and t[-1] > t[0] and q[0] > 0:
             d_guess = -np.log(max(q[-1], 0.01)/q[0]) / (t[-1] - t[0])
             d_guess = max(d_guess, 1e-6)
        else:
             d_guess = 0.001

        p0 = [qi_guess, d_guess]
        bounds = ([1e-5, 1e-9], [np.inf, np.inf])
        
        popt, pcov = curve_fit(exponential_decline, t, q, p0=p0, bounds=bounds, maxfev=10000)
        
        y_fit = exponential_decline(t, *popt)
        metrics = calculate_metrics(q, y_fit, n_params=2)
        
        return {
            "model": "Exponential",
            "params": popt.tolist(),
            "metrics": metrics,
            "fitted_flux": y_fit,
            "success": True,
            "error": None
        }
    except Exception as e:
        logger.warning(f"Exponential fit failed: {e}")
        return {"model": "Exponential", "success": False, "error": str(e)}

def fit_harmonic(t: np.ndarray, q: np.ndarray) -> Dict[str, Any]:
    """Fit Harmonic Decline."""
    try:
        qi_guess = np.max(q)
        d_guess = 0.001
        p0 = [qi_guess, d_guess]
        bounds = ([1e-5, 1e-9], [np.inf, np.inf])
        
        popt, pcov = curve_fit(harmonic_decline, t, q, p0=p0, bounds=bounds, maxfev=10000)
        
        y_fit = harmonic_decline(t, *popt)
        metrics = calculate_metrics(q, y_fit, n_params=2)
        
        return {
            "model": "Harmonic",
            "params": popt.tolist(),
            "metrics": metrics,
            "fitted_flux": y_fit,
            "success": True,
            "error": None
        }
    except Exception as e:
        logger.warning(f"Harmonic fit failed: {e}")
        return {"model": "Harmonic", "success": False, "error": str(e)}

def fit_hyperbolic(t: np.ndarray, q: np.ndarray) -> Dict[str, Any]:
    """Fit Hyperbolic Decline."""
    try:
        qi_guess = np.max(q)
        d_guess = 0.001
        b_guess = 0.5
        p0 = [qi_guess, d_guess, b_guess]
        # Bounds: qi>0, d>0, 0 < b <= 2
        bounds = ([1e-5, 1e-9, 1e-5], [np.inf, np.inf, 2.0])
        
        popt, pcov = curve_fit(hyperbolic_decline, t, q, p0=p0, bounds=bounds, maxfev=10000)
        
        y_fit = hyperbolic_decline(t, *popt)
        metrics = calculate_metrics(q, y_fit, n_params=3)
        
        return {
            "model": "Hyperbolic",
            "params": popt.tolist(),
            "metrics": metrics,
            "fitted_flux": y_fit,
            "success": True,
            "error": None
        }
    except Exception as e:
        logger.warning(f"Hyperbolic fit failed: {e}")
        return {"model": "Hyperbolic", "success": False, "error": str(e)}

def fit_all_models(t: np.ndarray, q: np.ndarray, metric: str = "rmse") -> Dict[str, Any]:
    """
    Fit all models and select the best one based on metric.
    Metric options: 'rmse', 'aic' (lower is better for both).
    """
    results = {}
    results["Exponential"] = fit_exponential(t, q)
    results["Harmonic"] = fit_harmonic(t, q)
    results["Hyperbolic"] = fit_hyperbolic(t, q)
    # Calculate Ensemble (Weighted Average based on AIC weights)
    # Weight w_i = exp(-0.5 * delta_AIC_i) / sum(...)
    # For simplicity, we just average the predictions of successful models for now, 
    # but to fit into "best_model_data" structure we need parameters.
    # Actually, Ensemble is a "Meta-Model". It doesn't have simple parameters.
    # We will perform ensemble logic IF requested, or just consider it another 'result'.
    
    # Let's add an "Ensemble" entry to results
    successful_models = [m for m, d in results.items() if d["success"]]
    if successful_models:
        # Simple average of parameters? No, that's physically wrong.
        # Ensemble must be generated at Forecast time.
        # But we can store the "weights" as parameters?
        # Let's simple create a dummy successful result for Ensemble if we want to select it.
        # However, passing it to 'model_function' in app.py would fail.
        # So we skip adding it to 'results' here and handle it in app.py or a wrapper?
        # Better: Add a flag in results so app.py knows.
        pass

    # Select best model based on metric
    best_model_name = None
    best_metric_val = float('inf')
    
    for name, data in results.items():
        if not data['success']:
            continue
        val = data['metrics'][metric]
        if val < best_metric_val:
            best_metric_val = val
            best_model_name = name
            
    if best_model_name is None:
        return {"success": False, "error": "All models failed to fit.", "results": results}
        
    return {
        "success": True,
        "best_model_name": best_model_name,
        "best_model_data": results[best_model_name],
        "all_results": results
    }

from sklearn.ensemble import IsolationForest

def clean_data_advanced(t: np.ndarray, q: np.ndarray, contamination: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses Isolation Forest to detect and remove outliers.
    Returns (t_clean, q_clean, mask_kept)
    """
    if len(q) < 50:
        # Too small for reliable ML outlier detection, return as is
        return t, q, np.ones(len(q), dtype=bool)
        
    # IsolationForest expects 2D array
    X = q.reshape(-1, 1)
    
    # Fit
    clf = IsolationForest(contamination=contamination, random_state=42)
    y_pred = clf.fit_predict(X)
    
    # y_pred: 1 for inliers, -1 for outliers
    mask = y_pred == 1
    
    return t[mask], q[mask], mask

def fit_all_models_advanced(t: np.ndarray, q: np.ndarray, metric: str = "rmse", auto_clean: bool = False) -> Dict[str, Any]:
    """
    Enhanced fitting that optionally cleans data first.
    """
    t_fit, q_fit = t, q
    cleaning_info = {"performed": False, "removed_count": 0}
    
    if auto_clean:
        t_clean, q_clean, mask = clean_data_advanced(t, q)
        removed = len(q) - len(q_clean)
        cleaning_info = {"performed": True, "removed_count": removed}
        t_fit, q_fit = t_clean, q_clean
        
    result = fit_all_models(t_fit, q_fit, metric)
    result["cleaning_info"] = cleaning_info
    return result
