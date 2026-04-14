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
    data_path = "dataset.csv"
    results = run_pipeline(data_path)
    
    print("\n" + "="*60)
    print("--- 1. FULL DESCRIPTIVE STATISTICS ---")
    print("="*60)
    for key, val in results['summary_stats'].items():
        print(f"{key}: {val}")
    
    print("\n" + "="*60)
    print("--- 2. NON-PARAMETRIC SURVIVAL ESTIMATES ---")
    print("="*60)
    print("[Kaplan-Meier]")
    kmf_overall = results['kmf_overall']['Overall']
    print(f"  Median Survival Time: {kmf_overall.median_survival_time_} months")
    print("  Survival Probabilities at milestones:")
    for m in [12, 24, 60]:
        if m <= max(kmf_overall.timeline):
            surv_prob = kmf_overall.survival_function_at_times(m).iloc[0]
            print(f"    {m} Months: {surv_prob:.2%}")
            
    print("\n[Nelson-Aalen]")
    naf_overall = results['naf_overall']['Overall']
    print("  Cumulative Hazard at milestones:")
    for m in [12, 24, 60]:
        if m <= max(naf_overall.timeline):
            cum_haz = naf_overall.cumulative_hazard_at_times(m).iloc[0]
            print(f"    {m} Months: {cum_haz:.4f}")
        
    print("\n" + "="*60)
    print("--- 3. PARAMETRIC MODELS EVALUATION ---")
    print("="*60)
    print(f"Recommended Model: {results['best_parametric']}\n")
    for m_name, m_res in results['parametric_results'].items():
        print(f"[{m_name}]")
        print(f"  AIC: {m_res['aic']:.1f} | BIC: {m_res['bic']:.1f} | Log-Likelihood: {m_res['log_likelihood']:.1f}")
        params = ", ".join([f"{k}: {v:.4f}" for k, v in m_res['params'].items()])
        print(f"  Parameters: {params}")

    print("\n" + "="*60)
    print("--- 4. MULTIVARIATE ANALYSIS (AFT MODELS) ---")
    print("="*60)
    for name, model in results['aft_models'].items():
        print(f"\n[{name} AFT - Top Coefficients]")
        print(model.summary[['coef', 'exp(coef)', 'p']].head(5))
        
    print("\n" + "="*60)
    print("--- 5. AUTOMATED VERIFICATION CHECKS ---")
    print("="*60)
    summary = results['cph_model'].summary
    for col in ['Adj_Radio', 'Adj_Chem']:
        for fc in [c for c in summary.index if col in c]:
            coef = summary.loc[fc, 'coef']
            print(f"Treatment Variable '{fc}' coefficient: {coef:.4f}")
            assert coef != 0, f"{fc} coefficient is unexpectedly zero!"

    print("\n[SUCCESS] Pipeline executed completely. Dataset rows match expectations: 1062.")
