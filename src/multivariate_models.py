from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
import pandas as pd
from typing import Dict, Any

def fit_cox_ph(df: pd.DataFrame, duration_col: str, event_col: str) -> CoxPHFitter:
    """
    Fits a Cox Proportional Hazards model.
    """
    cph = CoxPHFitter()
    # Need to handle categorical vars (dummies) before fitting if not already handled
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph

def fit_aft_models(df: pd.DataFrame, duration_col: str, event_col: str) -> Dict[str, Any]:
    """
    Fits various Accelerated Failure Time models.
    """
    models = {
        'Weibull AFT': WeibullAFTFitter(),
        'Log-Normal AFT': LogNormalAFTFitter(),
        'Log-Logistic AFT': LogLogisticAFTFitter()
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(df, duration_col=duration_col, event_col=event_col)
            results[name] = model
        except Exception as e:
            print(f"Failed to fit {name}: {e}")
            
    return results
