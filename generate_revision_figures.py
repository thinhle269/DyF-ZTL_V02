"""Generate the revision figures from the results_revision/ data.
All figures are drawn from real logged runs (no synthetic/stylized data):
  Fig_Convergence   - mean +/- std accuracy band over 10 seeds, 100 rounds, 3 methods
  Fig_Robustness    - 8 methods x 10 poison ratios (mean of 3 seeds; std band for top-2)
  Fig_TrustEvolution- real trust trajectories, DyF-ZTL seed 0 poison 0.2 (ground-truth
                      poisoned clients marked red) -> replaces the inconsistent old Fig.3/Fig.8
  Fig_Sensitivity   - hyperparameter sweep, accuracy at poison 0.2 / 0.4
  Fig_Performance   - clean-data metric bars with std error bars (10 seeds)
Output: results_revision/figures_paper/
"""
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'

OUT = "results_revision/figures_paper"
os.makedirs(OUT, exist_ok=True)

METHOD_ORDER = ['DyF-ZTL', 'FLTrust', 'FedProx', 'FedAvg', 'MultiKrum', 'Krum', 'Median', 'TrimmedMean']
COLORS = {'DyF-ZTL': '#d62728', 'FLTrust': '#1f77b4', 'FedProx': '#2ca02c', 'FedAvg': '#7f7f7f',
          'MultiKrum': '#9467bd', 'Krum': '#8c564b', 'Median': '#e377c2', 'TrimmedMean': '#bcbd22'}

# ---------- 1. Convergence (10-seed band) ----------
def fig_convergence():
    plt.figure(figsize=(9, 5.5))
    for method in ['FedAvg', 'FedProx', 'DyF-ZTL']:
        files = sorted(glob.glob(f"results_revision/main/per_round_{method}_seed*_p0.0.csv"))
        if not files:
            continue
        curves = np.stack([pd.read_csv(f)['Accuracy'].values for f in files])
        rounds = np.arange(curves.shape[1])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        c = COLORS[method]
        plt.plot(rounds, mean, label=f"{method} (n={len(files)})", color=c,
                 linewidth=2.2 if method == 'DyF-ZTL' else 1.6)
        plt.fill_between(rounds, mean - std, mean + std, color=c, alpha=0.15)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.ylim(80, 100)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig_Convergence_10seeds.png", dpi=300)
    plt.close()

# ---------- 2. Robustness (8 methods) ----------
def fig_robustness():
    pivot = pd.read_csv("results_revision/analysis/robust_pivot.csv", index_col=0)
    pivot_std = pd.read_csv("results_revision/analysis/robust_pivot_std.csv", index_col=0)
    ratios = [float(c) for c in pivot.columns]
    plt.figure(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        if method not in pivot.index:
            continue
        mean = pivot.loc[method].values.astype(float)
        c = COLORS[method]
        lw = 2.6 if method in ('DyF-ZTL', 'FLTrust') else 1.4
        ls = '-' if method in ('DyF-ZTL', 'FLTrust') else '--'
        plt.plot(ratios, mean, marker='o', markersize=5 if lw > 2 else 4,
                 label=method, color=c, linewidth=lw, linestyle=ls)
        if method in ('DyF-ZTL', 'FLTrust'):
            std = pivot_std.loc[method].values.astype(float)
            plt.fill_between(ratios, mean - std, mean + std, color=c, alpha=0.15)
    plt.xlabel("Poisoning Ratio")
    plt.ylabel("Global Test Accuracy (%)")
    plt.xticks(ratios)
    plt.ylim(-3, 100)
    plt.legend(ncol=2, loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig_Robustness_8methods.png", dpi=300)
    plt.close()

# ---------- 3. Trust evolution (real data, ground-truth attackers marked) ----------
def fig_trust():
    path = "results_revision/robust/trust_DyF-ZTL_seed0_p0.2.csv"
    df = pd.read_csv(path)
    rng = np.random.RandomState(0)                      # replicates fl_core poison selection
    poisoned = set(rng.choice(range(20), 4, replace=False).tolist())
    plt.figure(figsize=(9, 5.5))
    plt.plot(df['Round'], df['Threshold'], color='black', linestyle='--', linewidth=2.4,
             label=r'Adaptive threshold $\tau_r$')
    lab_p, lab_h = False, False
    for i in range(20):
        col = f'Client_{i}'
        if i in poisoned:
            plt.plot(df['Round'], df[col], color='#d62728', linewidth=2.0, alpha=0.9,
                     label='Poisoned clients (ground truth)' if not lab_p else None)
            lab_p = True
        else:
            plt.plot(df['Round'], df[col], color='#2ca02c', linewidth=1.0, alpha=0.45,
                     label='Honest clients' if not lab_h else None)
            lab_h = True
    plt.axvspan(-0.3, 2.5, color='gray', alpha=0.12)
    plt.text(1.1, 0.5, 'grace\nperiod', ha='center', fontsize=10, color='dimgray')
    plt.xlabel("Communication Round")
    plt.ylabel("Trust Score")
    plt.ylim(-0.03, 1.05)
    plt.legend(loc='center right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig_TrustEvolution_real.png", dpi=300)
    plt.close()

# ---------- 4. Sensitivity ----------
def fig_sensitivity():
    df = pd.read_csv("results_revision/analysis/sensitivity_pivot.csv", header=[0, 1], index_col=0)
    means = df['mean']
    stds = df['std']
    order = [v for v in means.sort_values(by=means.columns[0]).index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, ratio in zip(axes, means.columns):
        vals = means.loc[order, ratio]
        errs = stds.loc[order, ratio]
        colors = ['#d62728' if v < 94 else '#1f77b4' for v in vals]
        ax.barh(range(len(order)), vals, xerr=errs, color=colors, alpha=0.85, capsize=3)
        base = means.loc['baseline', ratio]
        ax.axvline(base, color='black', linestyle='--', linewidth=1.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlim(80, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"Poisoning ratio {ratio}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig_Sensitivity.png", dpi=300)
    plt.close()

# ---------- 5. Clean performance bars ----------
def fig_performance():
    summ = pd.read_csv("results_revision/analysis/main_summary.csv", index_col=0)
    methods = ['FedAvg', 'FedProx', 'DyF-ZTL']
    metrics = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.26
    plt.figure(figsize=(10, 5.5))
    for i, method in enumerate(methods):
        means = [summ.loc[method, f"{m} (mean)"] for m in metrics]
        stds = [summ.loc[method, f"{m} (std)"] for m in metrics]
        c = {'FedAvg': '#7f7f7f', 'FedProx': '#2ca02c', 'DyF-ZTL': '#d62728'}[method]
        bars = plt.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
                       label=method, color=c, alpha=0.85)
        for b, mval in zip(bars, means):
            plt.annotate(f"{mval:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 4),
                         textcoords='offset points')
    plt.xticks(x, metrics)
    plt.ylabel("Score (%)")
    plt.ylim(60, 105)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{OUT}/Fig_Performance_10seeds.png", dpi=300)
    plt.close()

fig_convergence()
fig_robustness()
fig_trust()
fig_sensitivity()
fig_performance()
print("[DONE] Figures written to", OUT)
