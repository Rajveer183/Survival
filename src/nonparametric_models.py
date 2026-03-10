from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test
import pandas as pd
from typing import Dict, Any, List

def fit_kaplan_meier(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str = None) -> Dict[str, Any]:
    """
    Fits Kaplan-Meier estimator(s).
    """
    results = {}
    if group_col:
        groups = df[group_col].unique()
        for g in groups:
            kmf = KaplanMeierFitter()
            mask = (df[group_col] == g)
            kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col], label=str(g))
            results[str(g)] = kmf
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(df[duration_col], df[event_col], label='Overall')
        results['Overall'] = kmf
    return results

def fit_nelson_aalen(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str = None) -> Dict[str, Any]:
    """
    Fits Nelson-Aalen estimator(s).
    """
    results = {}
    if group_col:
        groups = df[group_col].unique()
        for g in groups:
            naf = NelsonAalenFitter()
            mask = (df[group_col] == g)
            naf.fit(df.loc[mask, duration_col], df.loc[mask, event_col], label=str(g))
            results[str(g)] = naf
    else:
        naf = NelsonAalenFitter()
        naf.fit(df[duration_col], df[event_col], label='Overall')
        results['Overall'] = naf
    return results

def compare_groups(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str):
    """
    Performs log-rank test between groups.
    """
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    
    res = multivariate_logrank_test(df[duration_col], df[group_col], df[event_col])
    return res
