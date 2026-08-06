# DyF-ZTL: Federated Dynamic Fuzzy Learning for Privacy-Preserving Zero Trust Endpoint Security

Code, experiment logs, and paper artifacts for the manuscript *"DyF-ZTL: Federated Dynamic
Fuzzy Learning for Privacy-Preserving Zero Trust Endpoint Security"*. This repository
accompanies the revised submission: every number in the paper can be traced to the run
logs included here, and every experiment can be re-executed from the scripts below.

## Repository layout

```
├── src/
│   ├── preprocessing.py     # ToN_IoT loading, cleaning, 70/15/15 split, RF top-15
│   │                        # feature selection, non-IID Dirichlet partitioning, SMOTE
│   ├── models.py            # DeepNet (baseline MLP 15-64-32-7) and DynamicFuzzyNet (DFNN)
│   ├── trust_engine.py      # Adaptive Trust Engine (Algorithm 2 of the paper, verbatim)
│   ├── fl_core.py           # FL training loop: FedAvg / FedProx / trust filtering / defenses
│   ├── aggregators.py       # Krum, Multi-Krum, coordinate Median, Trimmed-Mean, FLTrust
│   └── utils.py             # metrics (macro P/R/F1, macro FPR, benign FAR) and plotting
├── run_experiments.py       # experiment driver (all presets, seeded, resume-safe)
├── analyze_results.py       # mean±std tables, paired t / Wilcoxon / Cohen's d_z, pivots
├── benchmark_timing.py      # 10-round timing probe (extrapolation check)
├── benchmark_timing_full.py # full 100-round wall-clock, 3 repetitions (paper Table 2)
├── generate_revision_figures.py  # regenerates all paper figures from the logs
├── make_latex_tables.py     # regenerates LaTeX table bodies from the analysis CSVs
├── figures/                 # the exact figures used in the paper (300 dpi)
└── results/
    ├── runs.csv             # one row per completed run (all 500+ experiments)
    ├── analysis/            # aggregated tables backing every result in the paper
    ├── confusion_matrices_seed0/  # per-class counts shown in the paper (seed 0)
    ├── trust_logs/          # logged trust trajectories (source of the trust figure)
    └── timing_full.json     # measured 100-round wall-clock, 3 repetitions per method
```

## Environment

Python 3.13, PyTorch 2.8 (CUDA optional — experiments were run on an NVIDIA Quadro RTX 4000).

```bash
pip install -r requirements.txt
```

## Dataset

Download the **ToN_IoT Windows 10** dataset (`Train_Test_Windows_10.csv`, ~37 MB) from the
official UNSW source: https://research.unsw.edu.au/projects/toniot-datasets
and place it at `dataset/Train_Test_Windows_10.csv`. The pipeline removes the 15 `mitm`
records and 759 rows with non-numeric values, yielding 35,201 samples
(24,640 train / 5,280 validation / 5,281 test, stratified, `random_state=42`).

## Reproducing the paper

Each preset is seeded end-to-end: a seed fixes the data partition, model initialization,
and the compromised-client set, so all methods face identical conditions per seed.
Completed runs are recorded in `results/runs.csv` and skipped on re-run (resume-safe).

```bash
# Table 2 + Table 3 (main comparison, 10 seeds x 100 rounds)
python run_experiments.py --exp main

# Table 5 (8 methods x 10 poisoning ratios x 3 seeds)
python run_experiments.py --exp robust

# Table 6 (trusted-set-size analysis incl. FLTrust with full D_val)
python run_experiments.py --exp valsize

# Table 7 (DFNN + robust aggregation rules)
python run_experiments.py --exp fuzzyagg

# Tables 8-9 (ablation, clean + under attack)
python run_experiments.py --exp ablation
python run_experiments.py --exp ablation_poison

# Table 10 (hyperparameter sensitivity)
python run_experiments.py --exp sensitivity

# Table 2 timing column (run on an otherwise idle machine)
python benchmark_timing_full.py

# Aggregate + statistics, then figures and LaTeX tables
python analyze_results.py
python generate_revision_figures.py
python make_latex_tables.py
```

Approximate cost on a single workstation GPU: the full suite is ~500 runs
(≈2–3 GPU-days sequentially; the driver supports multiple concurrent processes,
which coordinate through claim files and per-run result markers).

## Mapping paper results to logs

| Paper artifact | Source |
|---|---|
| Table 2 (performance, 10 seeds) | `results/analysis/main_summary.csv`, `results/timing_full.json` |
| Table 3 (significance, CI) | `results/analysis/main_tests_*.csv` |
| Table 4 / confusion figures | `results/confusion_matrices_seed0/` |
| Table 5 (robustness) | `results/analysis/robust_pivot.csv` (+`_std`) |
| Table 6 (trusted-set size) | `results/analysis/valsize_summary.csv` |
| Table 7 (DFNN + rules) | `results/analysis/fuzzyagg_pivot.csv` |
| Tables 8–9 (ablation) | `results/analysis/main_summary.csv`, `results/analysis/ablation_poison_summary.csv` |
| Table 10 (sensitivity) | `results/analysis/sensitivity_pivot.csv` |
| Trust-evolution figure | `results/trust_logs/trust_DyF-ZTL_seed0_p0.2.csv` |
| Raw record of every run | `results/runs.csv` |

 
