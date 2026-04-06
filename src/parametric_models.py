from lifelines import (
    WeibullFitter, 
    ExponentialFitter, 
    LogNormalFitter, 
    LogLogisticFitter,
    GeneralizedGammaFitter,
    PiecewiseExponentialFitter
)
import pandas as pd
import numpy as np
from typing import Dict, Any

def fit_all_parametric(df: pd.DataFrame, duration_col: str, event_col: str) -> Dict[str, Any]:
    """
    Fits multiple parametric survival distributions.
    """
    fitters = {
        'Exponential': ExponentialFitter(),
        'Weibull': WeibullFitter(),
        'Log-Normal': LogNormalFitter(),
        'Generalized Gamma': GeneralizedGammaFitter(),
        'Piecewise Exponential': PiecewiseExponentialFitter(breakpoints=[12, 24, 36])
    }
    
    results = {}
    
    T = df[duration_col]
    E = df[event_col]
    
    for name, fitter in fitters.items():
        try:
            fitter.fit(T, E)
            results[name] = {
                'fitter': fitter,
                'aic': fitter.AIC_,
                'bic': fitter.BIC_,
                'log_likelihood': fitter.log_likelihood_,
                'params': fitter.params_.to_dict()
            }
        except Exception as e:
            print(f"Failed to fit {name}: {e}")
            
    return results

def get_best_parametric_model(parametric_results: Dict[str, Any]) -> str:
    """
    Identifies the best model based on AIC.
    """
    if not parametric_results:
        return "None"
    return min(parametric_results, key=lambda x: parametric_results[x]['aic'])
