import pandas as pd
import os
from typing import Optional

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the survival dataset from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    return df
