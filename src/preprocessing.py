import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans data, handles missing values, and identifies column roles.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe.
        Dict: Metadata containing column roles (duration, event, covariates, etc.).
    """
    df = df.copy()
    
    # Drop ID column if exists
    if 'ID_REF' in df.columns:
        df = df.drop(columns=['ID_REF'])
    
    # Detect duration and event columns
    duration_col = None
    event_col = None
    
    duration_keywords = ['time', 'duration', 'months', 'days', 'dfs', 'os', 'surv']
    event_keywords = ['event', 'status', 'death', 'dead', 'cens', 'dfs_event']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in duration_keywords) and pd.api.types.is_numeric_dtype(df[col]):
            duration_col = col
            break
            
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in event_keywords) and pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's binary
            if df[col].nunique() <= 2:
                event_col = col
                break
    
    # Fallback if detection fails (specific to this dataset)
    if not duration_col: duration_col = 'DFS (in months)'
    if not event_col: event_col = 'DFS event'
    
    # Clean missing values
    df = df.dropna()
    
    # Identify covariates
    all_cols = list(df.columns)
    covariates = [c for c in all_cols if c not in [duration_col, event_col]]
    
    categorical_cols = [c for c in covariates if df[c].dtype == 'object' or df[c].nunique() < 10]
    numerical_cols = [c for c in covariates if c not in categorical_cols]
    
    # Type conversion
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        
    metadata = {
        'duration_col': duration_col,
        'event_col': event_col,
        'covariates': covariates,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    
    return df, metadata

def encode_categorical(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encodes categorical variables for modeling.
    """
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
