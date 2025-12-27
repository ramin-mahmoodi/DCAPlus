import pandas as pd
import numpy as np

# -- 1. Cloud Connector (Mock) --
class CloudConnector:
    def connect(self, service: str):
        return True
        
    def query(self, sql: str):
        # Return dummy data dataframe
        return pd.DataFrame({"Result": ["Connected to " + sql]})
        
# -- 2. Type Curve Normalization --
def normalize_type_curve(wells_data: list):
    """
    wells_data: list of dicts {'t': np.array, 'q': np.array}
    Returns normalized t and q arrays.
    """
    normalized = []
    for w in wells_data:
        peak_q = np.max(w['q'])
        peak_idx = np.argmax(w['q'])
        
        # Shift time so peak is at t=0 (or t=1)
        t_shifted = w['t'] - w['t'][peak_idx]
        t_norm = t_shifted[t_shifted >= 0]
        q_norm = w['q'][t_shifted >= 0] / peak_q
        
        normalized.append({'t': t_norm, 'q': q_norm})
    return normalized

# -- 3. AI Report Generator --
def generate_narrative_report(
    well_name: str,
    eur: float,
    model_name: str,
    downtime_days: int
) -> str:
    """
    Template-based NLP generation.
    """
    return f"""
    EXECUTIVE SUMMARY FOR {well_name.upper()}
    
    Physical Analysis:
    The production behavior is best characterized by the {model_name} decline model, suggesting a specific flow regime.
    
    Operational Issues:
    The well experienced approximately {downtime_days} days of downtime/shut-ins, which heavily impacted the short-term cash flow.
    
    Reserves Estimation:
    Our AI-driven forecast predicts an EUR of {eur:,.0f} barrels.
    
    Recommendation:
    Review pump efficiency during the identified downtime periods.
    """
