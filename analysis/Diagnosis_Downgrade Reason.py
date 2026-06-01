"""
诊断：为什么所有3态基因都被降级为2态？
"""
import pandas as pd
import numpy as np

results = pd.read_csv('plan_b_state_driven_results.csv')
data = pd.read_csv('MEF_QC_all.csv', index_col=0)

print("=" * 70)
print("诊断：3态基因被降级的原因")
print("=" * 70)

# 检查3态聚类的基因特征
three_state_genes = results[results['dominant_state'] == 3]

print(f"\n3态聚类基因总数: {len(three_state_genes)}")

# 统计降级原因
reasons = {
    'low_fano': 0,          # Fano < 15
    'low_separation': 0,    # ksyn2/ksyn1 < 2
    'low_p2': 0,            # p2 < 0.05
    'multiple_reasons': 0
}

for idx, row in three_state_genes.iterrows():
    gene_name = row['gene']
    if gene_name not in data.index:
        continue
    
    gene_data = data.loc[gene_name].values
    gene_data = gene_data[~np.isnan(gene_data)]
    
    mean_expr = np.mean(gene_data)
    var_expr = np.var(gene_data)
    fano = var_expr / mean_expr if mean_expr > 0 else 0
    
    ksyn1 = row['ksyn1']
    ksyn2 = row['ksyn2']
    p2 = row['p2']
    
    separation = ksyn2 / (ksyn1 + 1e-10)
    
    # 判断降级原因
    triggered = []
    if fano < 15:
        triggered.append('low_fano')
    if separation < 2.0:
        triggered.append('low_separation')
    if p2 < 0.05:
        triggered.append('low_p2')
    
    if len(triggered) == 1:
        reasons[triggered[0]] += 1
    elif len(triggered) > 1:
        reasons['multiple_reasons'] += 1

print("\n降级原因统计:")
for reason, count in reasons.items():
    if count > 0:
        print(f"  {reason}: {count}个基因 ({count/len(three_state_genes)*100:.1f}%)")

# 详细分析每个聚类
print("\n" + "=" * 70)
print("按聚类详细分析")
print("=" * 70)

for cluster_id in sorted(three_state_genes['cluster'].unique()):
    cluster_genes = three_state_genes[three_state_genes['cluster'] == cluster_id]
    
    print(f"\n聚类{cluster_id}: {len(cluster_genes)}个基因")
    
    fanos = []
    separations = []
    p2s = []
    
    for idx, row in cluster_genes.iterrows():
        gene_name = row['gene']
        if gene_name not in data.index:
            continue
        
        gene_data = data.loc[gene_name].values
        gene_data = gene_data[~np.isnan(gene_data)]
        
        mean_expr = np.mean(gene_data)
        var_expr = np.var(gene_data)
        fano = var_expr / mean_expr if mean_expr > 0 else 0
        
        ksyn1 = row['ksyn1']
        ksyn2 = row['ksyn2']
        p2 = row['p2']
        
        fanos.append(fano)
        separations.append(ksyn2 / (ksyn1 + 1e-10))
        p2s.append(p2)
    
    print(f"  Fano因子: 中位数={np.median(fanos):.1f}, 范围=[{np.min(fanos):.1f}, {np.max(fanos):.1f}]")
    print(f"  ksyn2/ksyn1: 中位数={np.median(separations):.2f}, 范围=[{np.min(separations):.2f}, {np.max(separations):.2f}]")
    print(f"  p2概率: 中位数={np.median(p2s):.3f}, 范围=[{np.min(p2s):.3f}, {np.max(p2s):.3f}]")
    
    # 判断主要原因
    low_fano_count = sum(1 for f in fanos if f < 15)
    low_sep_count = sum(1 for s in separations if s < 2.0)
    low_p2_count = sum(1 for p in p2s if p < 0.05)
    
    print(f"  触发降级规则:")
    print(f"    Fano<15: {low_fano_count}/{len(fanos)} ({low_fano_count/len(fanos)*100:.1f}%)")
    print(f"    分离度<2: {low_sep_count}/{len(separations)} ({low_sep_count/len(separations)*100:.1f}%)")
    print(f"    p2<0.05: {low_p2_count}/{len(p2s)} ({low_p2_count/len(p2s)*100:.1f}%)")

print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

print("\n可能的原因:")
print("  1. Fano<15的阈值太严格（应该降低到10-12）")
print("  2. 分离度<2的阈值太严格（应该降低到1.5）")
print("  3. p2<0.05过于严格（可能需要0.02-0.03）")
print("  4. 三个规则是'或'关系，太容易触发降级")

print("\n建议调整:")
print("  方案A: 放宽单个阈值")
print("    - Fano < 10 (而非15)")
print("    - 分离度 < 1.5 (而非2.0)")
print("    - p2 < 0.02 (而非0.05)")
print("\n  方案B: 改为'与'逻辑")
print("    - 需要同时满足多个条件才降级")
print("    - 例如: (Fano<12) AND (分离度<1.5)")
