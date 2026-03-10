from src.data_loader import load_data
from src.preprocessing import preprocess_data, encode_categorical
from src.descriptive_analysis import get_summary_statistics
from src.nonparametric_models import fit_kaplan_meier, fit_nelson_aalen
from src.parametric_models import fit_all_parametric, get_best_parametric_model
from src.multivariate_models import fit_cox_ph, fit_aft_models
from src.diagnostics import detect_hazard_shape
import pandas as pd
import pickle
import os

def run_pipeline(file_path: str):
    # 1. Load
    df_raw = load_data(file_path)
    
    # 2. Preprocess
    df, meta = preprocess_data(df_raw)
    
    # 3. Descriptive
    summary_stats = get_summary_statistics(df, meta['duration_col'], meta['event_col'])
    
    # 4. Preparation for Multivariate (dummy encoding)
    df_encoded = encode_categorical(df, meta['categorical_cols'])
    # Need to ensure duration and event cols are not encoded or lost
    # pd.get_dummies might rename columns, we need to find them again or preserve them
    
    # 5. Non-parametric
    kmf_overall = fit_kaplan_meier(df, meta['duration_col'], meta['event_col'])
    naf_overall = fit_nelson_aalen(df, meta['duration_col'], meta['event_col'])
    
    # 6. Parametric
    parametric_results = fit_all_parametric(df, meta['duration_col'], meta['event_col'])
    best_parametric = get_best_parametric_model(parametric_results)
    
    # 7. Multivariate
    cph_model = fit_cox_ph(df_encoded, meta['duration_col'], meta['event_col'])
    aft_models = fit_aft_models(df_encoded, meta['duration_col'], meta['event_col'])
    
    # 8. Hazard Shape
    hazard_shape = detect_hazard_shape(df, meta['duration_col'], meta['event_col'])
    
    results = {
        'df': df,
        'df_encoded': df_encoded,
        'meta': meta,
        'summary_stats': summary_stats,
        'kmf_overall': kmf_overall,
        'naf_overall': naf_overall,
        'parametric_results': parametric_results,
        'best_parametric': best_parametric,
        'cph_model': cph_model,
        'aft_models': aft_models,
        'hazard_shape': hazard_shape
    }
    
    return results

if __name__ == "__main__":
    data_path = "Colorectal Cancer Patient Data.csv"
    results = run_pipeline(data_path)
    print("\n--- Pipeline Analysis Results ---")
    print(f"Cohort Size: {results['summary_stats']['total_obs']}")
    print(f"Events: {results['summary_stats']['events']}")
    print(f"Hazard Shape: {results['hazard_shape']}")
    print("--- Pipeline Run Successfully ---")
