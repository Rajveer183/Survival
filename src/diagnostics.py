from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from typing import Dict, Any

def check_cox_assumptions(cph: CoxPHFitter, df: pd.DataFrame) -> Any:
    """
    Checks the proportional hazards assumption using Schoenfeld residuals.
    """
    return cph.check_assumptions(df, p_value_threshold=0.05)

def compute_residuals(cph: CoxPHFitter, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Computes different types of residuals for diagnostic plots.
    """
    # Martingale residuals
    martingale = cph.compute_residuals(df, 'martingale')
    # Deviance residuals
    deviance = cph.compute_residuals(df, 'deviance')
    # Schoenfeld residuals are obtained via cph.check_assumptions or manually
    
    return {
        'martingale': martingale,
        'deviance': deviance
    }

def detect_hazard_shape(df: pd.DataFrame, duration_col: str, event_col: str) -> str:
    """
    Heuristic to detect the shape of the hazard function.
    IFR: Increasing Failure Rate
    DFR: Decreasing Failure Rate
    Bathtub: Decreasing then increasing
    """
    # Simply using a Weibull fit as a proxy
    from lifelines import WeibullFitter
    wf = WeibullFitter().fit(df[duration_col], df[event_col])
    rho = wf.rho_
    
    if 0.9 <= rho <= 1.1:
        return "Constant (Exponential-like)"
    elif rho > 1.1:
        return "Increasing (IFR)"
    elif rho < 0.9:
        return "Decreasing (DFR)"
    else:
        return "Complex"
