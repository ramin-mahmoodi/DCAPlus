import numpy as np
import pandas as pd
from typing import List, Dict, Any

def optimize_portfolio(
    candidates: List[Dict[str, Any]],
    total_budget: float,
    strategy: str = "Max NPV"
) -> Dict[str, Any]:
    """
    Selects the best combination of wells to drill/complete/rework given a budget.
    This is a simplified Knapsack Problem.
    
    candidates: List of dicts, e.g. [{ "name": "Well A", "capex": 50000, "npv": 120000, "eur": 50000 }, ...]
    strategy: "Max NPV" or "Max EUR"
    """
    
    # 1. Calculate Profitability Index (PI) for ranking
    # PI = NPV / CAPEX
    
    metric_key = "npv" if strategy == "Max NPV" else "eur"
    
    ranked_wells = []
    for w in candidates:
        capex = w.get("capex", 1.0)
        val = w.get(metric_key, 0.0)
        pi = val / capex if capex > 0 else 0
        w["pi"] = pi
        ranked_wells.append(w)
        
    # Sort by PI descending (Greedy approach for Knapsack approx)
    ranked_wells.sort(key=lambda x: x["pi"], reverse=True)
    
    selected_wells = []
    current_spend = 0.0
    total_value = 0.0
    
    for well in ranked_wells:
        if current_spend + well["capex"] <= total_budget:
            selected_wells.append(well)
            current_spend += well["capex"]
            total_value += well[metric_key]
            
    # Remaining budget
    remaining = total_budget - current_spend
    
    return {
        "selected": selected_wells,
        "total_spend": current_spend,
        "total_value": total_value,
        "remaining_budget": remaining,
        "count": len(selected_wells)
    }
