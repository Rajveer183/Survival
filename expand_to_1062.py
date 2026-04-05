import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import re
import numpy as np

def expand_to_1062(source_path, target_path, target_rows=1062):
    print(f"Loading original dataset: {source_path}")
    df_original = pd.read_csv(source_path)
    # Always start from original 62 if possible, or just first 62
    df_original = df_original.iloc[:62].copy()
    
    # Detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_original)
    if 'ID_REF' in df_original.columns:
        metadata.update_column(column_name='ID_REF', sdtype='id')
    
    print("Training GaussianCopulaSynthesizer on original 62 rows...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_original)
    
    rows_needed = target_rows - len(df_original)
    print(f"Generating {rows_needed} synthetic rows with clinical logic...")
    synthetic_data = synthesizer.sample(num_rows=rows_needed)
    
    # --- ENFORCE CLINICAL LOGIC ON ALL DATA (Original + Synthetic) to make trends crystal clear ---
    df_combined = pd.concat([df_original, synthetic_data], ignore_index=True)
    
    # 1. Survival: D < C < B < A
    # 1. Survival: D < C < B < A
    stage_impact = {'A': 25, 'B': 10, 'C': -5, 'D': -15}
    # 2. Gender: Women > Men
    gender_impact = {'Female': 5, 'Male': 0}
    # 5. Location: Rectum < Colon < Left ~ Right
    # Handling user's 'anus' term as 'Colon' (or adjusting whatever is there)
    location_impact = {'Rectum': -10, 'Colon': -5, 'Left': 0, 'Right': 0}
    
    base_survival = 30
    
    # Recalculate DFS strictly
    df_combined['DFS (in months)'] = base_survival
    df_combined['DFS (in months)'] += df_combined['Dukes Stage'].map(stage_impact).fillna(0)
    df_combined['DFS (in months)'] += df_combined['Gender'].map(gender_impact).fillna(0)
    df_combined['DFS (in months)'] += df_combined['Location'].map(location_impact).fillna(0)
    
    # 3. Age: Increasing age -> decreasing survival
    df_combined['DFS (in months)'] -= (df_combined['Age (in years)'] - 50) * 0.3
    
    # 4. Treatment: Radiotherapy & Chemotherapy extends survival
    df_combined['DFS (in months)'] += df_combined['Adj_Radio'] * 8
    df_combined['DFS (in months)'] += df_combined['Adj_Chem'] * 12
    
    # Add realistic noise
    noise = np.random.normal(0, 15, len(df_combined))
    df_combined['DFS (in months)'] += noise
    
    # Clamp values to valid clinical bounds and make them whole numbers
    df_combined['DFS (in months)'] = df_combined['DFS (in months)'].clip(lower=1, upper=144).round().astype(int)
    
    # DFS event logic: Shorter survival -> highly likely an event (1)
    def calculate_event(surv_months):
        prob_event = 1.0 - (surv_months / 160)
        prob_event = np.clip(prob_event, 0.1, 0.95)
        return int(np.random.rand() < prob_event)
        
    df_combined['DFS event'] = df_combined['DFS (in months)'].apply(calculate_event)
    
    # Ensure IDs are correct logic
    # Synthetic ones start from GSM877189
    last_original_id = df_original['ID_REF'].iloc[-1]
    match = re.search(r'GSM(\d+)', last_original_id)
    start_num = int(match.group(1)) + 1 if match else 877189
    
    for i in range(len(df_original), len(df_combined)):
        df_combined.at[i, 'ID_REF'] = f"GSM{start_num + (i - len(df_original))}"
    
    print(f"Saving expanded dataset to: {target_path}")
    df_combined.to_csv(target_path, index=False)
    print(f"Success! {len(df_combined)} rows written with STRICT enforced clinical logic.")

if __name__ == "__main__":
    expand_to_1062('Colorectal Cancer Patient Data.csv', 'Colorectal Cancer Patient Data_new.csv')

if __name__ == "__main__":
    expand_to_1062('Colorectal Cancer Patient Data.csv', 'Colorectal Cancer Patient Data_new.csv')
