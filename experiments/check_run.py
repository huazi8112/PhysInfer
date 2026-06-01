import os
import pandas as pd
import numpy as np

paths = ['test_ssa_方案B_results.csv', '222/test_ssa_方案B_results.csv']
found = None
for p in paths:
    if os.path.exists(p):
        found = p
        break

with open('result_info.txt', 'w', encoding='utf-8') as f:
    if not found:
        f.write('Not found anywhere\n')
    else:
        df = pd.read_csv(found)
        f.write(f'Found at: {found}\n')
        f.write(f'Shape: {df.shape}\n')
        if 'true_kon' in df.columns:
            true_bs = df['true_ksyn'] / df['true_koff']
            true_bf = (df['true_kon'] * df['true_koff']) / (df['true_kon'] + df['true_koff'])
            pred_bs = df['ksyn'] / df['koff']
            pred_kon = df['kon']
            pred_koff = df['koff']
            pred_bf = (pred_kon * pred_koff) / (pred_kon + pred_koff)
            r_bs = np.corrcoef(true_bs, pred_bs)[0, 1]
            r_bf = np.corrcoef(true_bf, pred_bf)[0, 1]
            f.write(f'R_BS: {r_bs:.4f}\n')
            f.write(f'R_BF: {r_bf:.4f}\n')
