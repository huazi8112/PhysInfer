"""
测试修复后的自适应可视化功能
"""
import sys
sys.path.insert(0, '222')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import nbinom

# 导入修复后的函数
exec(open('222/方案B_状态数驱动架构.py', encoding='utf-8').read().split('if __name__')[0])

print("=" * 70)
print("测试基因级自适应模型选择")
print("=" * 70)

# 加载数据
results = pd.read_csv('plan_b_state_driven_results.csv')
data = pd.read_csv('MEF_QC_all.csv', index_col=0)

print(f"\n数据加载成功:")
print(f"  基因数: {len(results)}")
print(f"  表达矩阵: {data.shape}")

# 生成自适应可视化
print(f"\n生成自适应可视化...")
plot_fitting_comparison(
    results, 
    data.values, 
    data.index.tolist(), 
    {},  # cluster_models (not used in visualization)
    {},  # adaptive_analysis (not used in visualization)
    'plan_b_fitting_adaptive.png'
)

print("\n✅ 自适应可视化已保存: plan_b_fitting_adaptive.png")
print("\n对比两张图:")
print("  - plan_b_fitting_comparison.png (旧版: 强制聚类级模型)")
print("  - plan_b_fitting_adaptive.png (新版: 自适应基因级模型)")
