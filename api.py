from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

# Import DCA logic (assuming in same directory or installed)
# In production, we'd restructure, but here we just import locally
import sys
import os
sys.path.append(os.getcwd())

from dca.fit import fit_all_models_advanced
from dca.eur import calculate_eur

app = FastAPI(title="DCA-Plus API", version="4.0")

class ProductionData(BaseModel):
    t_days: List[float]
    rates: List[float]

class ForecastRequest(BaseModel):
    data: ProductionData
    q_econ: float = 10.0
    horizon_years: int = 20

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to DCA-Plus REST API"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    """
    Calculates best fit and EUR for given time-series data.
    """
    t = np.array(req.data.t_days)
    q = np.array(req.data.rates)
    
    if len(t) != len(q):
        raise HTTPException(status_code=400, detail="Length of time and rate arrays must match")
        
    # Fit
    fit_res = fit_all_models_advanced(t, q, metric="aic", auto_clean=True)
    
    if not fit_res["success"]:
        raise HTTPException(status_code=500, detail="Fitting failed for all models")
        
    best_name = fit_res["best_model_name"]
    params = fit_res["best_model_data"]["params"]
    
    # EUR
    eur_res = calculate_eur(best_name, params, req.q_econ, req.horizon_years*365)
    
    return {
        "best_model": best_name,
        "params": params,
        "eur_bbl": eur_res["eur"],
        "remaining_life_days": eur_res["t_econ"],
        "metrics": fit_res["best_model_data"]["metrics"]
    }

# Instructions to run:
# pip install fastapi uvicorn
# uvicorn api:app --reload
