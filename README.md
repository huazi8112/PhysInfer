# PhysInfer reproducibility code

This code package corresponds to the locked PhysInfer main text, supplementary materials, and response to reviewers. The archive retains the legacy single `code/` root directory and provides the model definitions, model-order identification, BS/BF inference, simulation validation, real-data analysis, uncertainty assessment, frozen results, and figure-generation scripts in one place.

All commands should be run from the extracted `code/` directory. Model rates are expressed relative to the mRNA degradation rate; productive burst frequency (BF) denotes the stationary probability flux into the ON state.

## 1. Final models and notation

### 1.1 2-state model

- States: `OFF <-> ON`.
- Rates: `kon` denotes `OFF -> ON`, `koff` denotes `ON -> OFF`, and `ksyn` is the transcription rate in the ON state.
- Stationary distribution: exact Poisson–Beta.
- `BS = ksyn / koff`.
- `productive BF = pi_OFF * kon = pi_ON * koff`.

### 1.2 Effective 3-state model

- States: `(OFF1, OFF2, ON) = (G0, G1, G2)`.
- Topology: `OFF1 <-> OFF2 <-> ON`.
- Rates: `k01, k10, k12, k21`.
- `BS = ksyn / k21`.
- `productive BF = pi_OFF2 * k12 = pi_ON * k21`.

3-state support indicates that, within the tested model family, adding an effective OFF state improves the description of the stationary count distribution; it does not uniquely identify a molecular refractory mechanism.

### 1.3 Generator matrix and stationary equations

`Q[i,j]` denotes the rate from source state `i` to destination state `j`, with every row summing to zero. The row stationary distribution satisfies `pi @ Q = 0`; the column probability-vector form of the CME uses `Q.T`. The joint promoter–transcript system obtains a stationary solution by automatically expanding `m_max`, and checks boundary mass, stationary residuals, likelihood, and the numerical stability of BS/BF.

### 1.4 Model-order calls

The primary calls use adjacent-order held-out predictive gain, a matched lower-order null, joint bootstrap, and abstention. Only units receiving reproducible held-out support are assigned stable 2-state or 3-state calls; all others remain `ambiguous`.

The secondary BIC-plus-veto analysis describes model-order preference only and does not overwrite the primary held-out calls. Its rules are:

```text
Delta_BIC = BIC_2state - BIC_3state
tau = 5.0
R_sep = max(k01,k10,k12,k21) / (min(k01,k10,k12,k21) + 1e-6)
gamma = 2.5
```

- `Delta_BIC <= 0`: 2-state preference.
- `0 < Delta_BIC < tau`: 2-state preference.
- `Delta_BIC >= tau` and `R_sep < gamma`: 2-state preference.
- `Delta_BIC >= tau` and `R_sep >= gamma`: 3-state preference.

The implementation is in `src/physinfer/bic_annotation.py` and `scripts/run_secondary_bic_annotation.py`; the input `primary_call` is preserved unchanged.

### 1.5 Final loss functions

Both branches use the average stationary NLL across cells as the principal term, with numerical floor `epsilon = 1e-5`:

```text
L_2 = L_NLL,2 + 0.01 L_zero + 0.01 L_shape + 0.01 L_burst
L_3 = L_NLL,3 + 0.01 L_zero + 0.01 L_tail
```

The zero-count anchor is computed on the log-probability scale; each applicable auxiliary term is multiplied by `0.01`. The `distribution-only` control retains stationary NLL only. The implementation is in `src/physinfer/losses.py`.

## 2. Directory structure

```text
code/
├─ src/physinfer/                 Core model and inference modules
├─ scripts/                       Entry points for experiments, analysis, plotting, and auditing
├─ tests/                         Unit tests for core conventions
├─ data/                          Processed real data and input examples
├─ frozen_results/                Results locked for the manuscript and gene-level results
├─ figures/manuscript/            Manuscript figure files
├─ figures/final_analysis/        Final analysis figures
├─ reproduced_figures/            Reproduced PNG/PDF figures
├─ RUN_RELEASE_CHECKS.py          One-click PyCharm validation entry point
├─ RUN_48_CV20_PYCHARM.py         Entry point for the 48-unit experiment
├─ RESULT_LOCK.json               Definitions of locked results
├─ VALIDATION_REPORT.json         Audit results
├─ RELEASE_MANIFEST.json          File manifest
├─ SHA256SUMS.txt                 File hashes
├─ pyproject.toml                 Python package configuration
└─ requirements.txt               Dependency list
```

The mapping from the legacy package structure is: `core -> src/physinfer`, `experiments/analysis -> scripts`, `results -> frozen_results`, and `visualization -> scripts/reproduce_figures.py + figures`.

## 3. Installation

Python 3.10 or later is required. Creating an isolated virtual environment is recommended; do not store research results inside `.venv`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Alternatively, run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The principal dependencies are NumPy, SciPy, pandas, Matplotlib, scikit-learn, PyTorch, and pytest.

For PyCharm, open the `code/` directory, select a Python 3.10+ interpreter, install the dependencies in the terminal, and then run `RUN_RELEASE_CHECKS.py`.

## 4. One-click validation

```bash
python RUN_RELEASE_CHECKS.py
```

This entry point sequentially runs the unit tests, stationary-model validation, locked-result audit, and figure reproduction. They can also be run individually:

```bash
python -m pytest -q
python scripts/validate_installation.py
python scripts/audit_bic_cluster_results.py
python scripts/audit_release.py
python scripts/reproduce_figures.py --results frozen_results --output reproduced_figures
python scripts/verify_release_hashes.py
```

The expected core-test outcome is `7 passed`, and `all_checks_passed` in `VALIDATION_REPORT.json` should be `true`.

## 5. Input data

### 5.1 Real count matrix

`data/processed_MEF_gene_by_cell.csv` is the processed MEF gene-by-cell UMI matrix: 2,162 genes and 413 allele-specific profiles. The first column contains gene names, and `-1` denotes missing observations; there are 40,131 such entries. The matrix was obtained from the public processed resource described by Luo et al. and was constructed from the C57 and CAST allele-resolved UMI profiles reported by Larsson et al.; the corresponding manuscript references are [14] and [41].

```bash
python scripts/prepare_real_data.py data/processed_MEF_gene_by_cell.csv --output check_real_data
```

Under the numerical preflight rules `n_finite >= 200` and `max_finite_count <= 300`, the held-out analysis has 2,135 eligible genes. This scope and the 2,137-gene secondary BIC analysis set below are parallel analysis branches: the former is used for held-out model-order evaluation, whereas the latter is used for the nominal BIC preferences in Fig. 5D and Fig. S1.

### 5.2 Eight clustered BIC gene-level result files

The raw results are located in:

```text
frozen_results/secondary_bic/gene_level_clusters/
  cluster_0_identification_results.csv
  cluster_1_identification_results.csv
  ...
  cluster_7_identification_results.csv
```

The CSV fields are:

| Field | Meaning |
|---|---|
| `gene_index` | Index within a cluster file; it is repeated across clusters and cannot be used as a global key |
| `gene_name` | Gene name; the unique identifier after merging the eight files |
| `predicted_state` | BIC-plus-veto model-order preference, taking the value 2, 3, or missing |
| `confidence` | Confidence field preserved by the original results program; it is 1.0 in all eight current files |

The eight files contain 142, 276, 574, 86, 194, 35, 488, and 366 rows, respectively, for 2,161 rows in total. `predicted_state` is missing for 24 rows; the remaining 2,137 rows comprise 1,765 2-state preferences (82.6%) and 372 3-state preferences (17.4%).

To audit and merge them:

```bash
python scripts/audit_bic_cluster_results.py \
  --output bic_all_clusters_with_unresolved.csv \
  --summary bic_all_clusters_summary.json
```

The merged table adds `cluster` and `source_file` columns while retaining the 24 unresolved records. When analyzing only the 2,137 valid results, filter on non-missing `predicted_state`; do not rewrite the original eight CSV files.

To recompute BIC preferences from new paired fitting results, the input table must contain at least:

```text
gene,n_cells,total_nll_2state,total_nll_3state,k01,k10,k12,k21,primary_call
```

```bash
python scripts/run_secondary_bic_annotation.py paired_fits.csv \
  --output secondary_bic_annotations.csv --scope all
```

An example is provided at `data/examples/secondary_bic_input_example.csv`. `--scope ambiguous-only` processes only records whose primary call is ambiguous.

## 6. Simulation validation

### 6.1 ROC for 400 independent parameter units

```bash
python scripts/run_roc_n400.py --output results_roc_n400 --seed 2026
```

Quick check:

```bash
python scripts/run_roc_n400.py --smoke --output smoke_roc
```

The full protocol contains 400 independent parameter units, including 240 2-state and 160 3-state units. Each observation condition contains 100 parameter units, and each unit undergoes 30 held-out repetitions. The locked ROC AUC is 0.978, with a unit-stratified 95% bootstrap interval of 0.965–0.989. A complete run is time-consuming; use `--resume` to continue an interrupted run.

### 6.2 48-unit validation with 20% extrinsic noise

The full study condition uses 500 cells, 30% capture efficiency, and `CV_ext = 0.20`:

```bash
python scripts/run_selective_panel_48_cv20.py \
  --output results_48_cv20 --n-cells 500 --eta 0.3 --extrinsic-cv 0.20
```

In PyCharm, run `RUN_48_CV20_PYCHARM.py`. Here, `SMALL_VALIDATION = False` executes the full experiment; setting it to `True` performs a run check only.

```bash
python scripts/run_selective_panel_48_cv20.py --smoke --output smoke_selective_cv20
```

Extrinsic noise is defined as a cell-to-cell lognormal multiplier acting on `k_syn` with mean 1; it is not the empirical CV of the count matrix. For the locked 0.80/0.20 policy, 27 of 48 units received stable calls, coverage was 56.25%, all 27 calls agreed with the prespecified valid labels, and the remaining 21 units remained ambiguous.

## 7. Real-data model-order analysis

### 7.1 Adjacent-order held-out analysis

```bash
python scripts/run_real_model_order.py \
  data/processed_MEF_gene_by_cell.csv \
  check_real_data/eligible_manifest_2135.csv \
  condition_matched_k2_null_gains.csv \
  --output real_model_order --n-genes 2135 --repeats 30 --eta 1.0 --seed 2026
```

`condition_matched_k2_null_gains.csv` must use the same cell count, capture settings, and training/test splits as the real-data analysis. The script saves intermediate results for each gene.

The frozen 473-gene model-selection subset comprises the first 473 genes in frozen manifest order that completed all 30 prespecified repetitions. The initial results are 223 stable 2-state, 22 initial 3-state, and 228 ambiguous calls.

### 7.2 70/30 split-matched confirmation

```bash
python scripts/confirm_initial_k3_70_30.py \
  initial_calls.csv confirmation_gains_70_30.csv split_matched_k2_null.csv \
  --output confirmed_model_order.csv --seed 2026
```

All 22 initial 3-state candidates enter confirmation; 20 are retained as formal 3-state and 2 return to ambiguous. The final results are 223 stable 2-state, 20 formal 3-state, and 230 ambiguous calls.

## 8. BS/BF kinetic inference

```bash
python scripts/infer_routed_kinetics.py \
  data/processed_MEF_gene_by_cell.csv \
  frozen_results/real_data/FINAL_MODEL_ORDER_ROLES_473.csv \
  --output routed_kinetics --epochs 2000 --seeds 101 202 303
```

The network reads 15-dimensional distribution features that are invariant to cell ordering. The 2-state branch outputs `p_on, k_off, k_syn` and reconstructs `k_on`; the 3-state branch outputs `k01, k10, k12, k21, BS` and reconstructs `k_syn = BS*k21`. The fit with the lowest stationary NLL is selected across multiple initializations. The frozen kinetic panel contains 243 resolved genes: 223 stable 2-state genes plus 20 formal 3-state genes.

## 9. Loss-function ablation

`counts.npy` is a `[units, cells]` array; an optional truth CSV is aligned with its rows.

```bash
python scripts/run_loss_ablation.py counts_k2.npy \
  --truth truth_k2.csv --order 2 --output loss_ablation_k2 \
  --epochs 2000 --seeds 101 202 303

python scripts/run_loss_ablation.py counts_k3.npy \
  --truth truth_k3.csv --order 3 --output loss_ablation_k3 \
  --epochs 2000 --seeds 101 202 303
```

The locked results for Table 1 in the main text are in `frozen_results/loss_ablation/Table1_loss_ablation.csv`.

## 10. 3-state BS/BF uncertainty

`genes.txt` contains one gene name per line:

```bash
python scripts/run_k3_bootstrap_uq.py \
  data/processed_MEF_gene_by_cell.csv genes.txt \
  --output k3_bootstrap_uq --bootstraps 100 --eta 1.0 --seed 2026
```

The frozen focused cohort contains 7 independently sampled 3-state genes, with 100 cell-bootstrap stationary refits per gene, for 700 refits in total. This set is not assumed to be nested within the 20 formal 3-state genes in the 473-gene model-selection subset.

## 11. Practical recovery boundary

Input fields:

```text
estimator,condition,p_on,true_bs,true_bf,estimated_bs,estimated_bf
```

```bash
python scripts/summarize_recovery_envelopes.py predictions.csv \
  --output recovery_summary --fold-tolerances 1.5 2.0 3.0
```

The output summarizes the simultaneous recovery limits for BS and productive BF by estimator, observation condition, and fold criterion. The `P_on` limit is a practical recovery envelope that depends on cell number, capture efficiency, and error criterion; it is not a universal constant.

## 12. Figure reproduction

```bash
python scripts/reproduce_figures.py --results frozen_results --output reproduced_figures
```

This command generates PNG/PDF figures, including selective reliability/coverage, loss-function ablation, the 473-gene model-order flow, the resolved BS/BF landscape, 1.5-fold recovery, and the 2,137-gene BIC-preference donut chart. Manuscript figures are in `figures/manuscript/`, and final analysis figures are in `figures/final_analysis/`.

## 13. Frozen-result index

| Content | Path |
|---|---|
| 400-unit ROC summary | `frozen_results/validation/roc_n400_summary.json` |
| 48-unit selective policy | `frozen_results/validation/selective_policy_48_units.csv` |
| Eight clustered BIC gene-level tables | `frozen_results/secondary_bic/gene_level_clusters/` |
| 2,137-gene BIC summary | `frozen_results/secondary_bic/full_cohort_preference_summary.csv` |
| Final roles for 473 genes | `frozen_results/real_data/FINAL_MODEL_ORDER_ROLES_473.csv` |
| BS/BF for 243 genes | `frozen_results/real_data/FINAL_LOWEST_NLL_NEURAL_ESTIMATES.csv` |
| UQ for 7 genes | `frozen_results/focused_K3_UQ/` |
| Loss-function ablation | `frozen_results/loss_ablation/Table1_loss_ablation.csv` |
| Recovery envelopes | `frozen_results/recovery/` |

## 14. File integrity and repackaging

```bash
python scripts/verify_release_hashes.py
python scripts/build_release_manifest.py
python scripts/build_archive.py --output ../PhysInfer_modified_code.zip
```

When updating a Git repository, use the contents of `code/` as the repository root and exclude `.venv/`, `__pycache__/`, `.pytest_cache/`, temporary smoke outputs, and IDE configuration. Store new results in a separate result directory and add them to `frozen_results/` only after confirmation.

## 15. Interpretation boundaries

- 2-state/3-state calls provide effective model-order support; they do not uniquely identify every molecular mechanism.
- BIC preference describes a secondary model-optimization tendency and does not overwrite held-out stable/ambiguous calls.
- A neural network cannot create independent kinetic information that is absent from a static stationary distribution.
- Finite intervals for BS/productive BF do not imply that all microscopic rates are uniquely identifiable.
- Cross-gene BS/BF distributions and correlations are descriptive results.

The locked main text and supplementary materials take precedence over historical wording in legacy result files.
