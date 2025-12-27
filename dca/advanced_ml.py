import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from typing import Dict, Any

# -- 1. Neural Network Forecasting --
def train_neural_forecast(t: np.ndarray, q: np.ndarray, horizon: int = 365) -> Dict[str, Any]:
    """
    Uses MLPRegressor to forecast production.
    Features: Time, Time^2, Log(Time)
    """
    # Feature Engineering
    X = np.column_stack([t, t**2, np.log(t+1)])
    y = q
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MLP
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500)
    model.fit(X_scaled, y)
    
    # Forecast
    t_future = np.arange(t[-1], t[-1] + horizon)
    X_future = np.column_stack([t_future, t_future**2, np.log(t_future+1)])
    X_future_scaled = scaler.transform(X_future)
    q_pred = model.predict(X_future_scaled)
    q_pred[q_pred < 0] = 0
    
    return {"t_future": t_future, "q_pred": q_pred, "model": model}

# -- 2. Waterflood CRM --
def analyze_waterflood(
    q_oil: np.ndarray,
    q_inj: np.ndarray,
) -> Dict[str, Any]:
    """
    Capacitance Resistance Model (CRM) simplified.
    q_oil(t) = A * q_inj(t-lag) + B
    """
    # Create lag
    lag = 30
    if len(q_inj) <= lag: return {}
    
    X = q_inj[:-lag].reshape(-1, 1)
    y = q_oil[lag:]
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    return {
        "connectivity": reg.coef_[0],
        "base_prod": reg.intercept_,
        "score": reg.score(X, y)
    }

# -- 3. Downtime Detection --
def detect_downtime(q: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Returns boolean mask where True = Downtime (Shut-in).
    """
    return q < threshold

# -- 4. Spatial Heatmap --
def generate_heatmap(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray, 
    grid_size: int = 50
):
    """
    Interpolates values (EUR) onto a grid.
    """
    grid_x, grid_y = np.mgrid[min(lons):max(lons):complex(grid_size), min(lats):max(lats):complex(grid_size)]
    grid_z = griddata((lons, lats), values, (grid_x, grid_y), method='cubic')
    
    return grid_x, grid_y, grid_z
