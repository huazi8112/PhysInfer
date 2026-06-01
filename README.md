# PhysInfer: An Unsupervised Framework for Adaptive Inference of Single-Cell Transcriptional Bursting Dynamics via Physics-Constrained Neural Networks
-------------------

![PhysInfer Graphical Abstract](images/figure1.png)

### Framework Overview

> **Figure 1: Schematic overview of the PhysInfer framework.** 
> **(a) Core Concept**: By integrating adaptive model selection and parameter decoupling within a physics-constrained neural network, it reparameterizes microscopic rates into an orthogonal macroscopic space of burst size and burst frequency, addressing parameter unidentifiability in static data.
> **(b) Algorithmic Pipeline**: Introduces a zero-inflated negative binomial (ZINB) module to decouple technical dropouts from biological zeros, and utilizes a physics-informed loss function incorporating cumulative distribution constraints and zero-count anchors for optimization inference.
> **(c)-(e) Performance Advantages**: Achieves high inference accuracy within the identifiability boundary governed by the ON-state probability ($P_{ON}$); its adaptive topology module successfully identifies three-state gene clusters with a deep refractory period, overcoming the degradation and misclassification issues of standard two-state fitting.

* * *

## 1. System Requirements

* **Environment**: Python 3.8+
* **Required Packages**: `torch` (for neural network construction and automatic differentiation), `pandas`, `numpy`, `scipy` (for mathematical calculations like distribution functions), `matplotlib`, `seaborn`, `scikit-learn`

* * *

## 2. Overview

Single-cell RNA sequencing (scRNA-seq) has established that gene expression is driven by stochastic transcriptional bursting. However, inferring burst dynamics from static scRNA-seq snapshots is fundamentally constrained by **parameter unidentifiability** and **model topology mismatch**.

To address this challenge, we propose **PhysInfer**, an unsupervised framework that integrates a physics-constrained neural network with adaptive model selection and parameter decoupling. A physics-informed loss function, incorporating cumulative distribution constraints and zero-count anchors, reparameterizes the optimization from unidentifiable microscopic rates to an orthogonal macroscopic space of burst size and burst frequency, thereby defining the physical identifiability boundary for static data inversion. Inference accuracy is governed by the ON-state probability ($P_{ON}$). The method resolves parameter degeneracy in low-to-medium regimes but encounters an unidentifiable boundary near saturation ($P_{ON} \rightarrow 1$), where it adaptively outputs uncertainty warnings and defaults to a constitutive expression model.

Applied to a mouse embryonic fibroblast (MEF) dataset, PhysInfer reconstructs a genome-wide transcriptional dynamic manifold, identifying burst frequency modulation as the dominant regulatory strategy. With the aid of a ZINB module and an adaptive topology module, it not only enhances robustness against technical noise but also uncovers a distinct cluster of **three-state** genes with a deep refractory period, which standard two-state fitting misclassifies or overlooks. Ultimately, PhysInfer provides a robust, topology-adaptive, and interpretable solution for deciphering dynamic transcriptional regulation from static single-cell transcriptomic snapshots.

* * *

## 3. Repository Structure

```text
Project_Root/
│
├── core/                               # Core algorithms and architecture modules
│   ├── PhysInfer.py           # [Core] Main PhysInfer network architecture and inference implementation
│   ├── method_v131.py                  # Historical versions or standard 2-state inference baselines
│   ├── update_loss.py                  # Physics-informed loss function (CDF constraints & zero-count anchors)
│   └── Description_Training Phase Gene-Level Selection Scheme.py     # Logical documentation for topology adaptation & parameter decoupling
│
├── analysis/                           # Mechanism analysis and comparative experiments
│   ├── Comparison_Cluster Level vs Gene Level.py           # Comparative validation of 3-state gene clusters vs. traditional clustering
│   ├── Analysis_Code vs. Thought Comparison.py           # Alignment analysis of network reparameterization vs. theoretical identifiability boundary
│   └── Diagnosis_Downgrade Reason.py                 # Divergence diagnosis and fallback mechanisms near saturation (P_ON -> 1)
│
├── experiments/                        # Experiment execution and evaluation scripts
│   ├── run_ablation.py                 # Independent ablation studies (verifying the necessity of ZINB, physics loss, etc.)
│   ├── check_run.py                    # Running status monitoring and inference accuracy validation
│   └── patch_sens.py                   # Boundary sensitivity testing across different implicit activation regimes (P_ON)
│
├── visualization/                      # Visualization and plotting scripts
│   ├── Test_Adaptive Visualization.py             # Adaptive 2D/3D manifold generation based on dynamic parameter distributions
│   ├── plot_Fig3B_from_planb.py        # Reproduces Fig3B (Genome-wide transcriptional dynamic manifold)
│   ├── plot_filtered_2state_scatter.py # Scatter plots for filtered genes in the low-to-medium activation regime
│   ├── plot_V96_2state_scatter.py      # Comparative scatter plots for specific 2-state fitting iterative versions
│   └── generate_all3state_plot.py      # Batch visualization for 3-state gene clusters with deep refractory periods
│
├── data/                               # Datasets and cache directory
│   ├── MEF_QC_all.csv                  # Quality-controlled (QC) real single-cell expression matrix of MEF
│   ├── *_200_2state_data.csv           # Synthetic validation data with Ground-Truth dynamic parameters
│   ├── benchmark_ground_truth.csv      # Ground Truth summary for benchmark testing
│   └── 800_genes_auc_cache.csv         # Cache for inference evaluation metrics (AUC/Error, etc.)
│
├── results/                            # Output directory for results
│   ├── ablation_*_results.csv          # Results comparison from multi-group network architecture ablation studies
│   ├── cluster_*_identification.csv    # Multi-state (e.g., 3-state) gene classification results identified by the adaptive topology module
│   └── plan_b_nn_eval_metrics.csv      # Final evaluation metrics log for the PhysInfer network
│
└── README.md                           # Project documentation
```

* * *

## 4. File Descriptions

### 4.1 Core Algorithms and Model Construction (`core/`)

| File | Description |
| --- | --- |
| `PhysInfer.py` | **Core Architecture** (The PhysInfer main network). Contains adaptive feed-forward neural networks that decouple the orthogonal space of burst size and frequency. |
| `update_loss.py` / `update_loss_kwargs.py`| **Physics-Informed Optimization**. Implements loss functions incorporating cumulative distribution function (CDF) constraints and zero-expression anchors. |
| `method_v131.py` | **Baseline Code**. Earlier version implementations used for direct comparison with standard 2-state models, corroborating the superiority of three-state gene discovery. |

### 4.2 Experimental Analysis and Validation (`analysis/` & `experiments/`)

| File | Description |
| --- | --- |
| `Comparison_Cluster Level vs Gene Level.py` | Used to extract representative key genes possessing specific topological features (e.g., deep refractory periods). |
| `Analysis_Code vs. Thought Comparison.py` | Verifies whether the physics reparameterization theory aligns perfectly with the current neural network gradient flows and update rules. |
| `run_ablation.py` | **Ablation Studies**. Strips the ZINB module or loss constraints to validate the proposed architecture's effectiveness in tackling dropouts and breaking degeneracy. |
| `Diagnosis_Downgrade Reason.py` / `patch_sens.py` | Detects or compensates for inference failures or active model degradation into Poisson expression at the blind spot boundary ($P_{ON} \rightarrow 1$). |

### 4.3 Plotting and Visualization (`visualization/`)

| File | Description |
| --- | --- |
| `plot_Fig3B_from_planb.py` | Reconstructs the genome-wide dynamic transcriptional manifold (Fig3B) from output data, visually demonstrating that burst frequency modulation dominates MEF regulation. |
| `generate_all3state_plot.py`| Automates batch exporting of scatter and fitting curves for three-state heterogeneous genes identified via topology adaptation, proving they are neither overlooked nor misclassified. |

* * *

## 5. Execution Pipeline

It is recommended to execute the experimental scripts in the following order to fully reproduce the PhysInfer validation and real-data inference:

```text
Step 1: python baseline/test scripts (if applicable) → Validate synthetic data accuracy using benchmark_ground_truth.csv
          ↓
Step 2: python PhysInfer.py              → Load MEF_QC_all.csv and complete genome-wide dynamics inference on high-dimensional data
          ↓
Step 3: python run_ablation.py                   → Execute ablation studies, outputting comparative data before/after stripping core modules (e.g., ZINB/physics constraints)
          ↓
Step 4: python plot_Fig3B_from_planb.py (etc.)   → Extract the generated CSVs to plot the macroscopic manifold distribution of bursting parameters and three-state feature maps
```

* * *

## 6. Notes

*   **Inference Uncertainty Warnings**: If the model outputs `uncertainty warnings` during iteration, it indicates that the corresponding gene distribution falls within the $P_{ON}$ saturation regime. By default, the system may fall back to a constitutive expression model, which is a normal network behavior.
*   **Cache Cleaning**: If you update the loss function design or basic preprocessing methods, please manually delete cache files like `800_genes_auc_cache.csv` to prevent interference during recalculation.
*   **GPU Acceleration**: When computing large tensors or fitting large matrices inside `PhysInfer.py`, the `torch` module will automatically utilize GPU acceleration if a valid CUDA environment is configured; monitoring VRAM usage is recommended to prevent memory overflow.

* * *
