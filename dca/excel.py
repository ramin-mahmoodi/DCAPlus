import pandas as pd
import io
import openpyxl

def generate_excel_export(
    forecast_df: pd.DataFrame,
    params: list,
    model_name: str,
    q_econ: float
) -> bytes:
    """
    Generates an Excel file with data and basic summary using openpyxl.
    """
    output = io.BytesIO()
    
    # Create Excel Writer
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Forecast
        forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
        
        # Sheet 2: Parameters
        param_dict = {
            "Model": [model_name],
            "qi": [params[0] if len(params)>0 else 0],
            "Di": [params[1] if len(params)>1 else 0],
            "b": [params[2] if len(params)>2 else 0],
            "q_econ": [q_econ]
        }
        pd.DataFrame(param_dict).to_excel(writer, sheet_name='Parameters', index=False)
        
    return output.getvalue()
