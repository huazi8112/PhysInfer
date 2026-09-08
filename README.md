# PhysInfer reproducibility code

本代码包对应锁定后的 PhysInfer 正文、补充材料和回复意见。压缩包沿用旧版的单一 `code/` 根目录，并集中提供模型定义、态数识别、BS/BF 推断、模拟验证、真实数据分析、不确定性评估、冻结结果和绘图脚本。

所有命令均应在解压后的 `code/` 目录运行。模型速率以 mRNA 降解率为参照；productive burst frequency（BF）表示进入 ON 状态的稳态概率通量。

## 1. 最终模型与符号约定

### 1.1 2-state 模型

- 状态：`OFF <-> ON`。
- 速率：`kon` 表示 `OFF -> ON`，`koff` 表示 `ON -> OFF`，`ksyn` 是 ON 状态转录速率。
- 稳态分布：exact Poisson–Beta。
- `BS = ksyn / koff`。
- `productive BF = pi_OFF * kon = pi_ON * koff`。

### 1.2 Effective 3-state 模型

- 状态：`(OFF1, OFF2, ON) = (G0, G1, G2)`。
- 拓扑：`OFF1 <-> OFF2 <-> ON`。
- 速率：`k01, k10, k12, k21`。
- `BS = ksyn / k21`。
- `productive BF = pi_OFF2 * k12 = pi_ON * k21`。

3-state 支持表示在所检验模型族内，增加一个有效 OFF 状态能够改善稳态计数分布的描述，不等价于唯一确定某种分子层面的 refractory mechanism。

### 1.3 生成矩阵与稳态方程

`Q[i,j]` 表示源状态 `i` 到目标状态 `j` 的速率，每一行之和为零。行稳态分布满足 `pi @ Q = 0`，列概率向量形式的 CME 使用 `Q.T`。联合启动子—转录本系统通过自动扩张 `m_max` 获得稳态解，并检查边界质量、稳态残差、似然和 BS/BF 的数值稳定性。

### 1.4 模型阶数判定

主要判定使用相邻阶数 held-out predictive gain、匹配低阶 null、联合 bootstrap 和 abstention。只有得到可重复 held-out 支持的单位才获得稳定 2-state 或 3-state 判定；其余保留为 `ambiguous`。

二级 BIC-plus-veto 分析只描述模型阶数偏好，不覆盖主要 held-out 判定。规则为：

```text
Delta_BIC = BIC_2state - BIC_3state
tau = 5.0
R_sep = max(k01,k10,k12,k21) / (min(k01,k10,k12,k21) + 1e-6)
gamma = 2.5
```

- `Delta_BIC <= 0`：2-state preference。
- `0 < Delta_BIC < tau`：2-state preference。
- `Delta_BIC >= tau` 且 `R_sep < gamma`：2-state preference。
- `Delta_BIC >= tau` 且 `R_sep >= gamma`：3-state preference。

实现位于 `src/physinfer/bic_annotation.py` 和 `scripts/run_secondary_bic_annotation.py`，输入的 `primary_call` 会被原样保留。

### 1.5 最终损失函数

两个分支均以逐细胞平均稳态 NLL 为主要项，数值下限为 `epsilon = 1e-5`：

```text
L_2 = L_NLL,2 + 0.01 L_zero + 0.01 L_shape + 0.01 L_burst
L_3 = L_NLL,3 + 0.01 L_zero + 0.01 L_tail
```

零计数锚点在对数概率尺度上计算；每个适用辅助项分别乘以 `0.01`。`distribution-only` 对照仅保留稳态 NLL。实现位于 `src/physinfer/losses.py`。

## 2. 目录结构

```text
code/
├─ src/physinfer/                 核心模型和推断模块
├─ scripts/                       实验、分析、绘图及审计入口
├─ tests/                         核心约定单元测试
├─ data/                          处理后的真实数据与输入示例
├─ frozen_results/                正文锁定结果和逐基因结果
├─ figures/manuscript/            稿件图文件
├─ figures/final_analysis/        最终分析图
├─ reproduced_figures/            重绘的 PNG/PDF 图
├─ RUN_RELEASE_CHECKS.py          PyCharm 一键验证入口
├─ RUN_48_CV20_PYCHARM.py         48 单元实验入口
├─ RESULT_LOCK.json               最终结果口径
├─ VALIDATION_REPORT.json         审计结果
├─ RELEASE_MANIFEST.json          文件清单
├─ SHA256SUMS.txt                 文件哈希
├─ pyproject.toml                 Python 包配置
└─ requirements.txt               依赖列表
```

旧包目录的对应关系为：`core -> src/physinfer`，`experiments/analysis -> scripts`，`results -> frozen_results`，`visualization -> scripts/reproduce_figures.py + figures`。

## 3. 安装

需要 Python 3.10 或更高版本。建议新建独立虚拟环境，不要把研究结果保存在 `.venv` 目录内。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

也可以执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

主要依赖为 NumPy、SciPy、pandas、Matplotlib、scikit-learn、PyTorch 和 pytest。

PyCharm 使用步骤：打开 `code/` 目录，选择 Python 3.10+ 解释器，在终端安装依赖，然后运行 `RUN_RELEASE_CHECKS.py`。

## 4. 一键验证

```bash
python RUN_RELEASE_CHECKS.py
```

该入口依次执行单元测试、稳态模型验证、锁定结果审计和图形重绘。也可以逐项运行：

```bash
python -m pytest -q
python scripts/validate_installation.py
python scripts/audit_bic_cluster_results.py
python scripts/audit_release.py
python scripts/reproduce_figures.py --results frozen_results --output reproduced_figures
python scripts/verify_release_hashes.py
```

预期核心测试为 `7 passed`，`VALIDATION_REPORT.json` 中 `all_checks_passed` 应为 `true`。

## 5. 输入数据

### 5.1 真实计数矩阵

`data/processed_GSE176044_gene_by_cell.csv` 是 gene-by-cell 矩阵：2,162 个基因、413 个细胞，第一列为基因名，`-1` 表示缺失观测，共 40,131 个。

```bash
python scripts/prepare_real_data.py data/processed_GSE176044_gene_by_cell.csv --output check_real_data
```

在 `n_finite >= 200` 且 `max_finite_count <= 300` 的数值预检查规则下，held-out 分析得到 2,135 个 eligible genes。该范围与下述 2,137 个 BIC 有效结果属于不同分析入口，不应直接互换。

### 5.2 八个分群 BIC 逐基因结果

原始结果位于：

```text
frozen_results/secondary_bic/gene_level_clusters/
  cluster_0_identification_results.csv
  cluster_1_identification_results.csv
  ...
  cluster_7_identification_results.csv
```

CSV 字段如下：

| 字段 | 含义 |
|---|---|
| `gene_index` | 分群文件内部索引；不同分群间会重复，不能作为全局主键 |
| `gene_name` | 基因名称；八个文件合并后的唯一标识 |
| `predicted_state` | BIC-plus-veto 模型阶数偏好，取值为 2、3 或缺失 |
| `confidence` | 原结果程序保存的置信字段；当前八个文件均为 1.0 |

八个文件的行数依次为 142、276、574、86、194、35、488 和 366，总计 2,161 行。其中 24 行的 `predicted_state` 缺失；其余 2,137 行包含 1,765 个 2-state preference（82.6%）和 372 个 3-state preference（17.4%）。

核查并合并：

```bash
python scripts/audit_bic_cluster_results.py \
  --output bic_all_clusters_with_unresolved.csv \
  --summary bic_all_clusters_summary.json
```

合并表会增加 `cluster` 和 `source_file` 两列，并保留 24 个未解析记录。只分析 2,137 个有效结果时，应按 `predicted_state` 非缺失筛选，不要改写原始八个 CSV。

如果从新的配对拟合结果重新计算 BIC 偏好，输入表至少包含：

```text
gene,n_cells,total_nll_2state,total_nll_3state,k01,k10,k12,k21,primary_call
```

```bash
python scripts/run_secondary_bic_annotation.py paired_fits.csv \
  --output secondary_bic_annotations.csv --scope all
```

示例位于 `data/examples/secondary_bic_input_example.csv`。`--scope ambiguous-only` 只处理主要判定为 ambiguous 的记录。

## 6. 模拟验证

### 6.1 400 个独立参数单元 ROC

```bash
python scripts/run_roc_n400.py --output results_roc_n400 --seed 2026
```

快速检查：

```bash
python scripts/run_roc_n400.py --smoke --output smoke_roc
```

完整协议包含 400 个独立参数单元，其中 240 个 2-state、160 个 3-state；每个观察条件包含 100 个参数单元，每个单元进行 30 次 held-out 重复。冻结结果的 ROC AUC 为 0.978，unit-stratified 95% bootstrap interval 为 0.965–0.989。完整运行耗时较长，可使用 `--resume` 续接。

### 6.2 48 单元、20% 外部噪声验证

完整研究条件为 500 个细胞、30% 捕获效率和 `CV_ext = 0.20`：

```bash
python scripts/run_selective_panel_48_cv20.py \
  --output results_48_cv20 --n-cells 500 --eta 0.3 --extrinsic-cv 0.20
```

PyCharm 可运行 `RUN_48_CV20_PYCHARM.py`。其中 `SMALL_VALIDATION = False` 执行完整实验；设为 `True` 只进行运行检查。

```bash
python scripts/run_selective_panel_48_cv20.py --smoke --output smoke_selective_cv20
```

外部噪声定义为作用于 `k_syn`、均值为 1 的细胞间 lognormal multiplier，不是计数矩阵的经验 CV。冻结 0.80/0.20 policy 结果为：48 个单元中稳定调用 27 个，coverage 为 56.25%，27/27 与预设有效标签一致，其余 21 个保留为 ambiguous。

## 7. 真实数据态数分析

### 7.1 相邻阶数 held-out 分析

```bash
python scripts/run_real_model_order.py \
  data/processed_GSE176044_gene_by_cell.csv \
  check_real_data/eligible_genes.csv \
  condition_matched_k2_null_gains.csv \
  --output real_model_order --n-genes 2135 --repeats 30 --eta 1.0 --seed 2026
```

`condition_matched_k2_null_gains.csv` 必须与真实分析使用相同的细胞数、捕获设置和训练/测试拆分。脚本会按基因保存中间结果。

冻结的 computationally completed cohort 包含完成全部 30 次预设重复的前 473 个基因；它是计算完成队列，不是新的生物学筛选层。初始结果为 223 stable 2-state、22 initial 3-state 和 228 ambiguous。

### 7.2 70/30 split-matched 确认

```bash
python scripts/confirm_initial_k3_70_30.py \
  initial_calls.csv confirmation_gains_70_30.csv split_matched_k2_null.csv \
  --output confirmed_model_order.csv --seed 2026
```

22 个初始 3-state 候选均进入确认，20 个保留为 formal 3-state，2 个转回 ambiguous。最终结果为 223 stable 2-state、20 formal 3-state 和 230 ambiguous。

## 8. BS/BF 动力学推断

```bash
python scripts/infer_routed_kinetics.py \
  data/processed_GSE176044_gene_by_cell.csv \
  frozen_results/real_data/FINAL_MODEL_ORDER_ROLES_473.csv \
  --output routed_kinetics --epochs 2000 --seeds 101 202 303
```

网络读取 15 维、与细胞排列无关的分布特征。2-state 分支输出 `p_on, k_off, k_syn` 并重建 `k_on`；3-state 分支输出 `k01, k10, k12, k21, BS` 并重建 `k_syn = BS*k21`。多次初始化中以最低稳态 NLL 选择拟合。冻结动力学面板包含 243 个已解析基因，即 223 stable 2-state 加 20 formal 3-state。

## 9. 损失函数对照

`counts.npy` 为 `[units, cells]` 数组；可选 truth CSV 与数组行对应。

```bash
python scripts/run_loss_ablation.py counts_k2.npy \
  --truth truth_k2.csv --order 2 --output loss_ablation_k2 \
  --epochs 2000 --seeds 101 202 303

python scripts/run_loss_ablation.py counts_k3.npy \
  --truth truth_k3.csv --order 3 --output loss_ablation_k3 \
  --epochs 2000 --seeds 101 202 303
```

主文 Table 1 的冻结结果位于 `frozen_results/loss_ablation/Table1_loss_ablation.csv`。

## 10. 3-state BS/BF 不确定性

`genes.txt` 每行一个基因名称：

```bash
python scripts/run_k3_bootstrap_uq.py \
  data/processed_GSE176044_gene_by_cell.csv genes.txt \
  --output k3_bootstrap_uq --bootstraps 100 --eta 1.0 --seed 2026
```

冻结 focused cohort 包含 7 个独立抽样的 3-state 基因，每个基因 100 次 cell-bootstrap stationary refit，共 700 次。该集合与完成队列中的 20 个 formal 3-state 不假定嵌套关系。

## 11. Practical recovery boundary

输入字段：

```text
estimator,condition,p_on,true_bs,true_bf,estimated_bs,estimated_bf
```

```bash
python scripts/summarize_recovery_envelopes.py predictions.csv \
  --output recovery_summary --fold-tolerances 1.5 2.0 3.0
```

输出按 estimator、observation condition 和 fold criterion 汇总 BS 与 productive BF 的同时恢复上限。`P_on` 上限是依赖细胞数、捕获效率和误差判据的 practical recovery envelope，不是通用常数。

## 12. 图形重绘

```bash
python scripts/reproduce_figures.py --results frozen_results --output reproduced_figures
```

命令生成 PNG/PDF，包括 selective reliability/coverage、损失对照、473 基因阶数流程、resolved BS/BF landscape、1.5-fold recovery 和 2,137 基因 BIC preference 环形图。稿件图保存在 `figures/manuscript/`，最终分析图保存在 `figures/final_analysis/`。

## 13. 冻结结果索引

| 内容 | 路径 |
|---|---|
| 400 单元 ROC 汇总 | `frozen_results/validation/roc_n400_summary.json` |
| 48 单元 selective policy | `frozen_results/validation/selective_policy_48_units.csv` |
| 八个 BIC 逐基因分群表 | `frozen_results/secondary_bic/gene_level_clusters/` |
| BIC 全队列汇总 | `frozen_results/secondary_bic/full_cohort_preference_summary.csv` |
| 473 基因最终角色 | `frozen_results/real_data/FINAL_MODEL_ORDER_ROLES_473.csv` |
| 243 基因 BS/BF | `frozen_results/real_data/FINAL_LOWEST_NLL_NEURAL_ESTIMATES.csv` |
| 7 基因 UQ | `frozen_results/focused_K3_UQ/` |
| 损失对照 | `frozen_results/loss_ablation/Table1_loss_ablation.csv` |
| Recovery envelopes | `frozen_results/recovery/` |

## 14. 文件完整性和重新封包

```bash
python scripts/verify_release_hashes.py
python scripts/build_release_manifest.py
python scripts/build_archive.py --output ../PhysInfer_modified_code.zip
```

更新 Git 仓时，建议将 `code/` 内容作为仓库根目录，并排除 `.venv/`、`__pycache__/`、`.pytest_cache/`、临时 smoke 输出和 IDE 配置。新结果应保存在独立结果目录，经确认后再加入 `frozen_results/`。

## 15. 解释边界

- 2-state/3-state 是有效模型阶数支持，不是所有分子机制的唯一鉴定。
- BIC preference 表示二级模型优化倾向，不覆盖 held-out stable/ambiguous 判定。
- 神经网络不能创造静态稳态分布中不存在的独立动力学信息。
- BS/productive BF 的有限区间不代表所有微观速率均可唯一辨识。
- 跨基因 BS/BF 分布和相关性属于描述性结果。

锁定后的正文和补充材料定义优先于旧结果文件中的历史性文字说明。
