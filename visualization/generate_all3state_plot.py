import os
import pandas as pd
import numpy as np
import importlib.util

# 加载 bsbf 作为模块以重用内部函数
bsbf_path = r'c:\Users\zengxiaodong\Desktop\华子哥的任务\111\222\bsbf版本.py'
spec = importlib.util.spec_from_file_location('bsbf', bsbf_path)
bsbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsbf)

results_path = r'c:\Users\zengxiaodong\Desktop\华子哥的任务\111\plan_b_state_driven_results.csv'
data_path = r'c:\Users\zengxiaodong\Desktop\华子哥的任务\111\MEF_QC_all （带基因名称）.csv'

out_png = r'c:\Users\zengxiaodong\Desktop\华子哥的任务\111\plan_b_all3state_fitting.png'

N = 15  # 默认选择前 N 个 3 态基因

print('Loading results...')
res = pd.read_csv(results_path, encoding='utf-8-sig')
print('Loading data...')
data_df = pd.read_csv(data_path)
if 'gene_name' in data_df.columns:
    gene_names = data_df['gene_name'].tolist()
    data_values = data_df.drop('gene_name', axis=1).values.astype(np.float32)
else:
    data_df = pd.read_csv(data_path, index_col=0)
    gene_names = data_df.index.tolist()
    data_values = data_df.values.astype(np.float32)

# filter 3-state
three_df = res[res['dominant_state'] == 3].copy()
if three_df.empty:
    print('No 3-state genes found in results.')
    raise SystemExit(1)

# choose sort column
sort_col = 'kl_divergence'
if sort_col not in three_df.columns or three_df[sort_col].isna().all():
    sort_col = 'js_divergence'

if sort_col in three_df.columns:
    three_sorted = three_df.sort_values(by=sort_col, ascending=True)
else:
    three_sorted = three_df

selected = three_sorted.head(N)
print(f'Selected {len(selected)} 3-state genes (top {N} by {sort_col})')

selected_genes = []
for _, row in selected.iterrows():
    gname = row.get('gene_name')
    try:
        gid = gene_names.index(gname)
    except Exception:
        gid = int(row.get('gene_id', -1))
    rec = row.to_dict()
    rec['gene_id'] = gid
    rec['gene_name'] = gname
    # ensure k01/k10/k12/k21 present as floats or NaN
    for c in ['k01','k10','k12','k21','ksyn','ksyn2','ksyn_raw','BF','BS','pi','p_on']:
        if c in rec:
            try:
                rec[c] = float(rec[c])
            except Exception:
                rec[c] = np.nan
    selected_genes.append(rec)

# Now plot using adapted plotting logic from bsbf.plot_fitting_comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import nbinom

n_genes = len(selected_genes)
n_cols = 3
n_rows = (n_genes + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, gene_info in enumerate(selected_genes):
    ax = axes[i]
    gene_name = gene_info.get('gene_name')
    gene_idx = gene_info.get('gene_id')
    if gene_idx is None or gene_name not in gene_names:
        ax.text(0.5,0.5,'Invalid gene', ha='center')
        continue
    gene_data = data_values[gene_idx, :]
    gene_data_clean = gene_data[~np.isnan(gene_data)]
    max_val = int(np.max(gene_data_clean))
    bins = np.arange(0, min(max_val + 10, 500))
    counts, bin_edges = np.histogram(gene_data_clean, bins=bins, density=False)
    counts_norm = counts / (counts.sum() + 1e-12)
    ax.bar(bins[:-1], counts_norm, width=1.0, alpha=0.6, color='skyblue', edgecolor='black', linewidth=0.5)

    # 3-state plotting using params from gene_info
    k1_p = gene_info.get('k01', gene_info.get('k1_p', 0.5))
    k1_m = gene_info.get('k10', gene_info.get('k1_m', 0.5))
    k2_p = gene_info.get('k12', gene_info.get('k2_p', 5.0))
    k2_m = gene_info.get('k21', gene_info.get('k2_m', 5.0))
    k_ini = gene_info.get('ksyn2', gene_info.get('ksyn', gene_info.get('ksyn_raw', 50.0)))

    try:
        p0, p1, p2, _ = bsbf.solve_steady_state_3state(k1_p, k1_m, k2_p, k2_m)
    except Exception:
        p0 = 0.0

    w_inflate = p0
    w_active = 1.0 - p0

    kdeg = 1.0
    r_nb = k2_p + 1e-6
    prob_nb = (k2_m + kdeg) / (k2_m + kdeg + k_ini + 1e-8)
    nb_pmf = nbinom.pmf(bins[:-1], r_nb, prob_nb)
    theory_probs = w_active * nb_pmf
    theory_probs[0] += w_inflate
    theory_probs = theory_probs / (theory_probs.sum() + 1e-12)
    ax.plot(bins[:-1], theory_probs, 'r-', linewidth=2.5, label='3-state M1 (ZINB)')

    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlabel('Expression Level')
    ax.set_ylabel('Probability')
    ax.set_title(f"{gene_name}")
    ax.legend()

# hide extra axes
for j in range(n_genes, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print('WROTE', out_png)
