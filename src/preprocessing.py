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
    
    # Rename columns to match requested names
    mapping = {
        'Age (in years)': 'Age',
        'DFS (in months)': 'DFS',
        'DFS event': 'DFS_event'
    }
    df = df.rename(columns=mapping)
    
    # Target columns
    duration_col = 'DFS'
    event_col = 'DFS_event'
    
    # Requested Feature List
    covariates = [
        'Age',
        'Dukes Stage',
        'Gender',
        'Location',
        'Adj_Radio',
        'Adj_Chem'
    ]
    
    # Ensure all exist in df, otherwise drop them from list
    covariates = [c for c in covariates if c in df.columns]
    
    # Final cleanup: drop everything else
    cols_to_keep = covariates + [duration_col, event_col]
    df = df[cols_to_keep].dropna()
    
    categorical_cols = ['Dukes Stage', 'Gender', 'Location', 'Adj_Radio', 'Adj_Chem']
    categorical_cols = [c for c in categorical_cols if c in covariates]
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
