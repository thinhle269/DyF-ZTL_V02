"""Emit LaTeX (booktabs) table bodies from results_revision/analysis/ so the
manuscript revision can paste them directly. Output: results_revision/latex/.
Timing column uses the solo benchmark (results_revision/timing_solo.json);
the contended per-run times in runs.csv are intentionally NOT used.
"""
import json
import os

import pandas as pd

A = "results_revision/analysis"
OUT = "results_revision/latex"
os.makedirs(OUT, exist_ok=True)

def ms(summ, method, metric, digits=2):
    m = summ.loc[method, f"{metric} (mean)"]
    s = summ.loc[method, f"{metric} (std)"]
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"

def write(name, content):
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f"[OK] {path}")

summ = pd.read_csv(f"{A}/main_summary.csv", index_col=0)
timing = {}
if os.path.exists("results_revision/timing_solo.json"):
    with open("results_revision/timing_solo.json") as fh:
        timing = json.load(fh)
TIME_KEY = {'FedAvg': 'fedavg', 'FedProx': 'fedprox', 'DyF-ZTL': 'fuzzy'}

# ---------- Table: main comparison (10 seeds) ----------
rows = []
for method in ['FedAvg', 'FedProx', 'DyF-ZTL']:
    label = "DyF-ZTL (Proposed)" if method == 'DyF-ZTL' else method
    cells = [label,
             ms(summ, method, 'Accuracy'),
             ms(summ, method, 'Detection Rate (Recall)'),
             ms(summ, method, 'Precision'),
             ms(summ, method, 'F1-Score'),
             ms(summ, method, 'False Alarm Rate (FAR)'),
             ms(summ, method, 'FAR Benign->Attack')]
    if timing:
        cells.append(f"{timing[TIME_KEY[method]]['extrapolated_100r_s']:.0f}")
    rows.append(" & ".join(cells) + r" \\")
timing_col = " & Time (s)$^\\dagger$" if timing else ""
tab_main = (
    "% Table: comparative performance, mean +/- std over 10 seeds, 100 rounds, clean data.\n"
    "% dagger: wall-clock for one full 100-round run measured on an idle machine\n"
    "% (10 measured rounds x10); statistical metrics come from the 10-seed runs.\n"
    "\\begin{table*}[t]\n\\centering\n"
    "\\caption{Comparative performance on ToN\\_IoT Windows 10 over 100 communication rounds "
    "(mean $\\pm$ std over 10 seeded runs; identical data partitions per seed).}\n"
    "\\label{tab:main}\n"
    "\\begin{tabular}{l" + "c" * (6 + (1 if timing else 0)) + "}\n\\toprule\n"
    "Method & Accuracy (\\%) & Macro Recall (\\%) & Macro Precision (\\%) & Macro F1 (\\%) & "
    "Macro FPR (\\%) & FAR (\\%)" + timing_col + r" \\" + "\n\\midrule\n"
    + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
write("tab_main.tex", tab_main)

# ---------- Table: significance ----------
sig_rows = []
for metric in ['Accuracy', 'F1-Score']:
    f = f"{A}/main_tests_{metric.replace(' ', '_')}.csv"
    if not os.path.exists(f):
        continue
    t = pd.read_csv(f)
    t = t[t['Comparison'].isin(['DyF-ZTL vs FedAvg', 'DyF-ZTL vs FedProx'])]
    for _, r in t.iterrows():
        sig_rows.append(f"{metric} & {r['Comparison'].replace('DyF-ZTL vs ', '')} & "
                        f"{r['n_seeds']} & {r['diff']:+.3f} & {r['t_p']:.4f} & "
                        f"{r['wilcoxon_p']:.4f} & {r['cohens_d']:.2f} \\\\")
tab_sig = (
    "\\begin{table}[t]\n\\centering\n"
    "\\caption{Paired significance tests for DyF-ZTL against each baseline over 10 common seeds "
    "(paired $t$-test and Wilcoxon signed-rank; Cohen's $d$ for paired samples).}\n"
    "\\label{tab:significance}\n"
    "\\begin{tabular}{llccccc}\n\\toprule\n"
    "Metric & vs. & $n$ & $\\Delta$ (pp) & $p_{t}$ & $p_{W}$ & $d$ \\\\\n\\midrule\n"
    + "\n".join(sig_rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
write("tab_significance.tex", tab_sig)

# ---------- Table: robustness ----------
pv = pd.read_csv(f"{A}/robust_pivot.csv", index_col=0)
order = ['FedAvg', 'FedProx', 'Krum', 'MultiKrum', 'Median', 'TrimmedMean', 'FLTrust', 'DyF-ZTL']
ratios = list(pv.columns)
lines = []
for m in order:
    if m not in pv.index:
        continue
    label = "\\textbf{DyF-ZTL (Proposed)}" if m == 'DyF-ZTL' else m
    vals = " & ".join(f"{pv.loc[m, r]:.2f}" for r in ratios)
    lines.append(f"{label} & {vals} \\\\")
tab_rob = (
    "\\begin{table*}[t]\n\\centering\n"
    "\\caption{Global accuracy (\\%) under label-flipping poisoning "
    "(mean over 3 seeds, 20 rounds per point; all methods share identical partitions, "
    "poisoned-client sets, and attack per seed; Krum/Multi-Krum/Trimmed-Mean receive the oracle "
    "number of attackers $f$).}\n"
    "\\label{tab:robustness}\n"
    "\\begin{tabular}{l" + "c" * len(ratios) + "}\n\\toprule\n"
    "Method & " + " & ".join(str(r) for r in ratios) + " \\\\\n\\midrule\n"
    + "\n".join(lines) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
write("tab_robust.tex", tab_rob)

# ---------- Table: ablation ----------
ab_rows = []
for method, desc in [('DyF-ZTL', 'Full framework'),
                     ('FuzzyNoTrust', 'w/o Trust Engine (fuzzy model only)'),
                     ('DeepTrust', 'w/o Fuzzy layer (MLP + Trust Engine)'),
                     ('FedAvg', 'w/o both (plain FedAvg, MLP)')]:
    n = int(summ.loc[method, 'Accuracy (count)'])
    ab_rows.append(f"{desc} & {ms(summ, method, 'Accuracy')} & "
                   f"{ms(summ, method, 'F1-Score')} & {n} \\\\")
tab_ab = (
    "\\begin{table}[t]\n\\centering\n"
    "\\caption{Ablation study (clean data, 100 rounds): both components contribute, and the "
    "Trust Engine additionally stabilizes training across seeds.}\n"
    "\\label{tab:ablation}\n"
    "\\begin{tabular}{lccc}\n\\toprule\n"
    "Configuration & Accuracy (\\%) & Macro F1 (\\%) & $n$ \\\\\n\\midrule\n"
    + "\n".join(ab_rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
write("tab_ablation.tex", tab_ab)

# ---------- Table: sensitivity ----------
sv = pd.read_csv(f"{A}/sensitivity_pivot.csv", header=[0, 1], index_col=0)
means, stds = sv['mean'], sv['std']
name_map = {
    'baseline': '\\textbf{Default} ($\\alpha{=}0.5$, $\\lambda{=}0.2$, gates $0.70/0.85$, floor $0.4$, warm-up $3$)',
    'alpha=0.25': '$\\alpha = 0.25$', 'alpha=1.0': '$\\alpha = 1.0$', 'alpha=1.5': '$\\alpha = 1.5$',
    'decay_factor=0.1': '$\\lambda = 0.1$', 'decay_factor=0.4': '$\\lambda = 0.4$', 'decay_factor=0.8': '$\\lambda = 0.8$',
    'low_gate=0.6': 'penalty gate $= 0.60$', 'low_gate=0.8': 'penalty gate $= 0.80$',
    'high_gate=0.8': 'recovery gate $= 0.80$', 'high_gate=0.9': 'recovery gate $= 0.90$',
    'min_safety_threshold=0.2': 'floor $\\tau_{\\min} = 0.2$', 'min_safety_threshold=0.6': 'floor $\\tau_{\\min} = 0.6$',
    'warmup_rounds=0': 'warm-up $= 0$ rounds', 'warmup_rounds=4': 'warm-up $= 4$ rounds',
}
sv_rows = []
for key in ['baseline', 'alpha=0.25', 'alpha=1.0', 'alpha=1.5', 'decay_factor=0.1', 'decay_factor=0.4',
            'decay_factor=0.8', 'low_gate=0.6', 'low_gate=0.8', 'high_gate=0.8', 'high_gate=0.9',
            'min_safety_threshold=0.2', 'min_safety_threshold=0.6', 'warmup_rounds=0', 'warmup_rounds=4']:
    if key not in means.index:
        continue
    c2 = f"{means.loc[key, means.columns[0]]:.2f} $\\pm$ {stds.loc[key, stds.columns[0]]:.2f}"
    c4 = f"{means.loc[key, means.columns[1]]:.2f} $\\pm$ {stds.loc[key, stds.columns[1]]:.2f}"
    sv_rows.append(f"{name_map.get(key, key)} & {c2} & {c4} \\\\")
tab_sv = (
    "\\begin{table}[t]\n\\centering\n"
    "\\caption{Sensitivity of DyF-ZTL to Trust-Engine hyperparameters (accuracy \\%, mean $\\pm$ std over "
    "3 seeds, 20 rounds, one parameter varied at a time). The framework is insensitive across wide ranges; "
    "only lowering the recovery gate to $0.80$ degrades performance.}\n"
    "\\label{tab:sensitivity}\n"
    "\\begin{tabular}{lcc}\n\\toprule\n"
    "Configuration & Poison $0.2$ & Poison $0.4$ \\\\\n\\midrule\n"
    + "\n".join(sv_rows) +
    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
write("tab_sensitivity.tex", tab_sv)

print("[DONE] LaTeX tables in", OUT)
