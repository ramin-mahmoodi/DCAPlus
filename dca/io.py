import pandas as pd
import numpy as np
import io
from typing import Dict, Any, Tuple, List

def load_and_qc_csv(file_buffer) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Reads CSV, performs QC, returns (clean_df_for_fit, full_df_processed, qc_report).
    full_df_processed includes outliers and non-positive points but sorted.
    clean_df_for_fit is strictly for fitting (positive rates, aggregated duplicates).
    """
    try:
        df = pd.read_csv(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")
        
    # Check columns
    required = {'date', 'oil_rate'}
    if not required.issubset(df.columns):
        # fuzzy matching could be added here, but requirement says "date, oil_rate"
        raise ValueError(f"CSV missing required columns. Found {df.columns}, needed {required}")
        
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    
    # Stats before cleaning
    n_raw = len(df)
    
    # 1. Handle Duplicates (aggregate mean)
    # Check if duplicates exist
    n_duplicates = df.duplicated(subset=['date']).sum()
    if n_duplicates > 0:
        # aggregate
        df = df.groupby('date', as_index=False)['oil_rate'].mean()
    
    # 2. Add time axis t_days
    if df.empty:
         raise ValueError("Dataframe empty after date parsing.")
         
    min_date = df['date'].min()
    df['t_days'] = (df['date'] - min_date).dt.total_seconds() / 86400.0
    
    # 3. Non-positive rate count
    n_nonpositive = (df['oil_rate'] <= 0).sum()
    
    # 4. Outlier Detection (Flag only) for the whole dataset
    # Rule: > 99.5 percentile
    q995 = df['oil_rate'].quantile(0.995)
    df['is_outlier'] = df['oil_rate'] > q995
    n_outliers = df['is_outlier'].sum()
    
    # 5. Create Fit Dataset (exclude <= 0)
    fit_df = df[df['oil_rate'] > 0].copy()
    n_fit = len(fit_df)
    
    qc_report = {
        "n_raw": n_raw,
        "n_fit": n_fit,
        "dropped_nonpositive": int(n_nonpositive),
        "duplicates_removed": int(n_duplicates),
        "outliers_flagged": int(n_outliers)
    }
    
    return fit_df, df, qc_report

def parse_batch_uploaded_files(uploaded_files: List[Any]) -> Dict[str, Any]:
    """
    Parses a list of uploaded files (from Streamlit) and returns a dict of results.
    """
    results = {}
    for file in uploaded_files:
        try:
            # Check if file has read attribute, else it might be bytes
            file.seek(0)
            fit_df, full_df, qc = load_and_qc_csv(file)
            results[file.name] = {
                "fit_df": fit_df,
                "qc": qc,
                "status": "success"
            }
        except Exception as e:
            results[file.name] = {
                "status": "failed",
                "error": str(e)
            }
    return results
