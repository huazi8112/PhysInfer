import os
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

ablation_names = {
    0: "Group 0 (Full Joint Model)",
    1: "Group 1 (w/o Fano physical constraint)",
    2: "Group 2 (w/o Zero & Burst anchor)",
    3: "Group 3 (w/o High-Order Moment)",
    4: "Group 4 (Pure NLL Model)"
}

print("| Ablation Group | BS Correlation (R) | BF Correlation (R) |")
print("| :------------- | :----------------: | :----------------: |")

def true_bf_flux_2state(kon, koff):
    pon = kon / (kon + koff + 1e-10)
    return pon * koff

true_df = pd.read_csv('222/test_ssa_combined_2old_3new_true_params.csv')

for group in range(5):
    csv_path = "test_ssa_方案B_results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    cmd = [
        "python", r"222\方案B_状态数驱动架构.py",
        "--test_ssa",
        "--data_file", "222/test_ssa_combined_2old_3new_data.csv",
        "--param_file", "222/test_ssa_combined_2old_3new_true_params.csv",
        "--ablation_group", str(group)
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"| {ablation_names[group]} | Error | Error |", flush=True)
        # print stderr for debug
        print(f"DEBUG STDERR for group {group}:\n{result.stderr}")
        continue
    
    if not os.path.exists(csv_path):
        print(f"| {ablation_names[group]} | N/A | N/A |", flush=True)
        continue
    
    pred_df = pd.read_csv(csv_path)
    df = pd.merge(pred_df, true_df, on='gene_name', suffixes=('_pred', '_true'), how='inner')
    
    # Assign True_BS/BF using original exact logic from plot script
    # Handle possible suffix variations from merge
    kon_col = 'true_kon_true' if 'true_kon_true' in df.columns else 'true_kon'
    koff_col = 'true_koff_true' if 'true_koff_true' in df.columns else 'true_koff'
    ksyn_col = 'true_ksyn_true' if 'true_ksyn_true' in df.columns else 'true_ksyn'
    
    if ksyn_col in df.columns and koff_col in df.columns:
        if 'True_BS' not in df.columns:
            df['True_BS'] = np.nan
        if 'True_BF' not in df.columns:
            df['True_BF'] = np.nan
            
        mask_2state = df['True_BS'].isna() & df[ksyn_col].notna()
        df.loc[mask_2state, 'True_BS'] = df.loc[mask_2state, ksyn_col] / (df.loc[mask_2state, koff_col] + 1e-10)
        df.loc[mask_2state, 'True_BF'] = true_bf_flux_2state(df.loc[mask_2state, kon_col], df.loc[mask_2state, koff_col])
    else:
        df['True_BS'] = df.get('burst_size_true', np.nan)
        df['True_BF'] = df.get('burst_frequency_true', np.nan)

    if 'burst_size_pred' in df.columns:
        df['Pred_BS'] = df['burst_size_pred']
    elif 'burst_size' in df.columns:
        df['Pred_BS'] = df['burst_size']
    elif 'BS' in df.columns:
        df['Pred_BS'] = df['BS']
        
    if 'burst_frequency_pred' in df.columns:
        df['Pred_BF'] = df['burst_frequency_pred']
    elif 'burst_frequency' in df.columns:
        df['Pred_BF'] = df['burst_frequency']
    elif 'BF' in df.columns:
        df['Pred_BF'] = df['BF']
        
    if 'actual_var' in df.columns:
        df['Obs_Fano'] = df['actual_var'] / (df['actual_mean'] + 1e-10)
    else:
        df['Obs_Fano'] = 1.0 

    df_2 = df[df['dominant_state'] == 2].copy() if 'dominant_state' in df.columns else df.copy()

    kon_col_2 = 'true_kon_true' if 'true_kon_true' in df_2.columns else ('true_kon' if 'true_kon' in df_2.columns else None)
    koff_col_2 = 'true_koff_true' if 'true_koff_true' in df_2.columns else ('true_koff' if 'true_koff' in df_2.columns else None)
    
    if kon_col_2 and koff_col_2:
        df_2['p_on'] = df_2[kon_col_2] / (df_2[kon_col_2] + df_2[koff_col_2] + 1e-10)
    else:
        df_2['p_on'] = 0.5

    valid_mask = (df_2['Obs_Fano'] >= 3.5) & (df_2['p_on'] <= 0.8)
    df_filtered = df_2[valid_mask].copy()
    df_filtered = df_filtered.dropna(subset=['True_BS', 'True_BF', 'Pred_BS', 'Pred_BF'])
    filtered_df = df_filtered[(df_filtered['True_BS']>0) & (df_filtered['True_BF']>0) & (df_filtered['Pred_BS']>0) & (df_filtered['Pred_BF']>0)]
    
    if len(filtered_df) > 10:
        r_bs, _ = pearsonr(np.log(filtered_df['True_BS']), np.log(filtered_df['Pred_BS']))
        r_bf, _ = pearsonr(filtered_df['True_BF'], filtered_df['Pred_BF'])
        print(f"| {ablation_names[group]} | {r_bs:.4f} | {r_bf:.4f} |", flush=True)
    else:
        print(f"| {ablation_names[group]} | N/A (len={len(filtered_df)}) | N/A |", flush=True)
    
    dest_path = f"ablation_group_{group}_results.csv"
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(csv_path, dest_path)
