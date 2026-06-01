import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set plot style to resemble the provided image
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_theme(style="whitegrid", context="notebook")

true_file = "identifiable_200_2state_params.csv"
pred_file = "ablation_group_0_results.csv"

# Fallback to the old random dataset if the new one isn't the target
if not pd.io.common.file_exists(true_file):
    true_file = "my_200_2state_params.csv"

true_df = pd.read_csv(true_file)
pred_df = pd.read_csv(pred_file)

# Merge
merged = pd.merge(pred_df, true_df, on='gene_name', suffixes=('_pred', '_true'), how='inner')

# Calculate Theoretical or Observed Fano
if 'actual_var' in merged.columns and 'actual_mean' in merged.columns:
    fano = merged['actual_var'] / (merged['actual_mean'] + 1e-10)
else:
    # Estimate Fano from true params
    fano = 1.0 + merged['burst_size_true'] * (1 - merged['burst_frequency_true']/merged['koff_true'])

# Calculate zero rate or Pon to filter "Poisson trap" genes
if 'true_kon' in merged.columns and 'true_koff' in merged.columns:
    p_on = merged['true_kon'] / (merged['true_kon'] + merged['true_koff'] + 1e-10)
elif 'kon_true' in merged.columns and 'koff_true' in merged.columns:
    p_on = merged['kon_true'] / (merged['kon_true'] + merged['koff_true'] + 1e-10)
else:
    p_on = np.zeros(len(merged)) + 0.5

# FILTER CONDITIONS
# "不符合要求的基因先剔除" (Filter out genes with Fano < 3.5 or Pon > 0.8)
valid_idx = (fano >= 3.5) & (p_on <= 0.8)

filtered_df = merged[valid_idx]
print(f"Total genes: {len(merged)}, Genes after filtering (Fano >= 3.5 & Pon <= 0.8): {len(filtered_df)}")

true_bs = filtered_df['burst_size_true']
pred_bs = filtered_df['burst_size_pred']

true_bf = filtered_df['burst_frequency_true']
pred_bf = filtered_df['burst_frequency_pred']
plot_fano = fano[valid_idx]

# Subplot configuration matching the style
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot BS
r_bs, _ = pearsonr(true_bs.fillna(0), pred_bs.fillna(0))
sc1 = axes[0].scatter(true_bs, pred_bs, c=plot_fano, cmap='viridis', alpha=0.7, edgecolors='grey', s=40)
min_bs, max_bs = min(true_bs.min(), pred_bs.min()), max(true_bs.max(), pred_bs.max())
axes[0].plot([min_bs*0.8, max_bs*1.2], [min_bs*0.8, max_bs*1.2], 'r--', zorder=1)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title(f"2-State BS Correlation (R={r_bs:.2f})", fontweight='bold')
axes[0].set_xlabel("True Effective BS")
axes[0].set_ylabel("Pred Effective BS")
axes[0].grid(True, alpha=0.5)

# Plot BF
r_bf, _ = pearsonr(true_bf.fillna(0), pred_bf.fillna(0))
sc2 = axes[1].scatter(true_bf, pred_bf, c=plot_fano, cmap='viridis', alpha=0.7, edgecolors='grey', s=40)
min_bf, max_bf = min(true_bf.min(), pred_bf.min()), max(true_bf.max(), pred_bf.max())
axes[1].plot([min_bf*0.8, max_bf*1.2], [min_bf*0.8, max_bf*1.2], 'r--', zorder=1)
axes[1].set_title(f"2-State BF Correlation (R={r_bf:.2f})", fontweight='bold')
axes[1].set_xlabel("True Burst Frequency (BF)")
axes[1].set_ylabel("Pred Burst Frequency (BF)")
axes[1].grid(True, alpha=0.5)

# Add colorbar
cbar = fig.colorbar(sc2, ax=axes, fraction=0.03, pad=0.04)
cbar.set_label('Log Fano Factor' if plot_fano.max() > 10 else 'Fano Factor')

plt.suptitle("Filtered 2-State Inference Validation (Fano >= 3.5)", y=1.02, fontsize=14, fontweight='bold')
plt.savefig("Filtered_2State_Scatter.pdf", dpi=300, bbox_inches='tight')
plt.savefig("Filtered_2State_Scatter.png", dpi=300, bbox_inches='tight')
print("Plots saved as Filtered_2State_Scatter.pdf and Filtered_2State_Scatter.png")
