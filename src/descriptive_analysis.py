import pandas as pd
import numpy as np
from typing import Dict

def get_summary_statistics(df: pd.DataFrame, duration_col: str, event_col: str) -> Dict:
    """
    Computes summary statistics for survival data.
    """
    total_obs = len(df)
    events = df[event_col].sum()
    censored = total_obs - events
    
    mean_surv = df[duration_col].mean()
    median_surv = df[duration_col].median()
    variance_surv = df[duration_col].var()
    
    event_proportion = (events / total_obs) * 100
    censoring_rate = (censored / total_obs) * 100
    
    return {
        'total_obs': total_obs,
        'events': events,
        'censored': censored,
        'mean_surv': round(mean_surv, 2),
        'median_surv': round(median_surv, 2),
        'variance_surv': round(variance_surv, 2),
        'event_proportion': round(event_proportion, 2),
        'censoring_rate': round(censoring_rate, 2)
    }

def get_distribution_data(df: pd.DataFrame, duration_col: str):
    """
    Prepares data for survival time distribution plots.
    """
    return df[duration_col]
