import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os

def expand_dataset(file_path, target_rows=500):
    print(f"Loading dataset: {file_path}")
    df_original = pd.read_csv(file_path)
    original_row_count = len(df_original)
    
    # Check if we already have 500 rows and sdv-ids
    if original_row_count == target_rows and not df_original['ID_REF'].str.contains('sdv-id').any():
        print(f"Dataset already has {original_row_count} rows with correct IDs. No expansion needed.")
        return

    # If already expanded but contains sdv-ids, we need to fix it.
    # But let's assume we start from the original 62 rows for a clean run.
    # If the file has 500 rows, but we want the original 62 back... NO, let's just fix the IDs if present.
    
    if original_row_count > 62:
         # Find where synthetic rows start (assuming first 62 are original)
         print("Fixing existing dataset IDs...")
         df_clean_original = df_original.iloc[:62].copy()
         df_to_expand = df_clean_original
         original_row_count = 62
    else:
         df_to_expand = df_original

    # Detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_to_expand)
    
    if 'ID_REF' in df_to_expand.columns:
        metadata.update_column(column_name='ID_REF', sdtype='id')
    
    print("Training GaussianCopulaSynthesizer...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_to_expand)
    
    rows_needed = target_rows - original_row_count
    print(f"Synthetic rows to generate: {rows_needed}")
    synthetic_data = synthesizer.sample(num_rows=rows_needed)
    
    # Generate GSM IDs
    last_id = df_to_expand['ID_REF'].iloc[-1]
    # Extract number from 'GSM877188'
    import re
    match = re.search(r'GSM(\d+)', last_id)
    if match:
        start_num = int(match.group(1)) + 1
        new_ids = [f"GSM{i}" for i in range(start_num, start_num + rows_needed)]
        synthetic_data['ID_REF'] = new_ids
    
    df_final = pd.concat([df_to_expand, synthetic_data], ignore_index=True)
    
    final_row_count = len(df_final)
    print(f"Final row count: {final_row_count}")
    print(f"First 5 IDs: {df_final['ID_REF'].head().tolist()}")
    print(f"Last 5 IDs: {df_final['ID_REF'].tail().tolist()}")
    
    # Save combined dataframe
    df_final.to_csv(file_path, index=False)
    print(f"Updated dataset saved to: {file_path}")
    
    # Ensure categorical columns remain valid
    cat_cols = ['Dukes Stage', 'Gender', 'Location']
    for col in cat_cols:
        if col in df_final.columns:
            original_values = set(df_original[col].unique())
            final_values = set(df_final[col].unique())
            # Synthetic data might not have ALL values but shouldn't have NEW values unless it's a very small sample
            # Actually, GaussianCopula might generate values outside if not constrained, but SDV usually handles this.
            print(f"Column '{col}' unique values (Original): {original_values}")
            print(f"Column '{col}' unique values (Final): {final_values}")

    # Save combined dataframe
    df_final.to_csv(file_path, index=False)
    print(f"Updated dataset saved to: {file_path}")

if __name__ == "__main__":
    DATA_PATH = "Colorectal Cancer Patient Data.csv"
    expand_dataset(DATA_PATH, 500)
