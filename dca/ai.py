import pandas as pd
from typing import Dict, Any

def query_copilot(user_query: str, context_data: Dict[str, Any]) -> str:
    """
    A rule-based 'AI' that answers questions about the production data.
    """
    query = user_query.lower()
    
    # Context
    best_model = context_data.get("best_model", "Unknown")
    eur = context_data.get("eur", 0)
    tecon = context_data.get("tecon", 0)
    npv = context_data.get("npv", 0)
    
    if "best model" in query or "algorithm" in query:
        return f"Based on AIC and RMSE, the **{best_model}** model fits this well best."
        
    if "eur" in query or "reserves" in query or "total oil" in query:
        return f"The estimated ultimate recovery (EUR) is **{eur:,.0f} barrels**."
        
    if "life" in query or "how long" in query or "date" in query:
        return f"The well is expected to reach its economic limit in **{tecon:,.0f} days**."
        
    if "npv" in query or "money" in query or "value" in query or "profit" in query:
        return f"The projected Net Present Value (NPV) is **${npv:,.2f}**."
        
    if "hello" in query or "hi" in query:
        return "Hello! I am your DCA-Plus Copilot. Ask me about EUR, NPV, or the Best Model."
        
    return "I'm not sure about that. Try asking about EUR, NPV, model type, or remaining life."
