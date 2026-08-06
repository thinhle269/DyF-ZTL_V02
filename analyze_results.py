"""
Statistical analysis for the revision experiments.

Reads results_revision/runs.csv and produces, per experiment:
  - mean +/- std tables across seeds (all metrics)
  - paired significance tests DyF-ZTL vs every other method on the same seeds
    (paired t-test + Wilcoxon signed-rank + Cohen's d for paired samples)
  - robustness pivot (method x poison ratio, mean accuracy)
  - sensitivity pivot (variant x poison ratio)

Usage:  python analyze_results.py [--outdir results_revision] [--metric Accuracy]
Outputs printed to stdout and written to <outdir>/analysis/.
Run it at any time; it uses whatever runs have finished so far.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

def load_runs(outdir):
    """Union runs.csv with the per-run final_*.json files, dedup, coerce numerics."""
    frames = []
    runs_csv = os.path.join(outdir, 'runs.csv')
    for path in [runs_csv] + sorted(glob.glob(runs_csv + '.part*.csv')):
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path, on_bad_lines='skip'))
            except Exception:
                pass
    for jf in glob.glob(os.path.join(outdir, '*', 'final_*.json')):
        try:
            with open(jf) as fh:
                frames.append(pd.DataFrame([json.load(fh)['metrics']]))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in METRICS + ['Seed', 'Poison_Ratio', 'Rounds']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.drop_duplicates(subset=['Experiment', 'Method', 'Seed', 'Poison_Ratio', 'Rounds'],
                            keep='first').reset_index(drop=True)
    return df

METRICS = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score',
           'False Alarm Rate (FAR)', 'FAR Benign->Attack', 'Training Time (s)']

def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else np.inf * np.sign(diff.mean() or 1)

def summarize(df, group_cols, out_path):
    present = [m for m in METRICS if m in df.columns]
    agg = df.groupby(group_cols)[present].agg(['mean', 'std', 'count'])
    agg.columns = [f"{m} ({s})" for m, s in agg.columns]
    agg = agg.round(3)
    agg.to_csv(out_path)
    return agg

def paired_tests(df, metric, proposed='DyF-ZTL', extra_group=None):
    rows = []
    group_cols = [extra_group] if extra_group else []
    for key, sub in (df.groupby(extra_group) if extra_group else [(None, df)]):
        base = sub[sub['Method'] == proposed].set_index('Seed')[metric]
        for method in sorted(sub['Method'].unique()):
            if method == proposed:
                continue
            other = sub[sub['Method'] == method].set_index('Seed')[metric]
            common = base.index.intersection(other.index)
            if len(common) < 2:
                continue
            a, b = base.loc[common].values, other.loc[common].values
            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:  # identical values
                w_stat, w_p = np.nan, 1.0
            rows.append({
                **({extra_group: key} if extra_group else {}),
                'Comparison': f"{proposed} vs {method}",
                'n_seeds': len(common),
                f'{proposed} mean': round(a.mean(), 3),
                f'other mean': round(b.mean(), 3),
                'diff': round((a - b).mean(), 3),
                't_p': round(t_p, 4), 'wilcoxon_p': round(w_p, 4),
                'cohens_d': round(cohens_d_paired(a, b), 3),
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_revision')
    ap.add_argument('--metric', default='Accuracy')
    args = ap.parse_args()

    df = load_runs(args.outdir)
    if df.empty:
        print(f"No results in {args.outdir} yet."); return
    analysis_dir = os.path.join(args.outdir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"Loaded {len(df)} runs "
          f"({', '.join(f'{e}:{n}' for e, n in df['Experiment'].value_counts().items())})\n")

    # --- main + ablation: clean-performance table with significance ---
    ma = df[df['Experiment'].isin(['main', 'ablation'])]
    if len(ma):
        print("=== MAIN + ABLATION (clean, mean +/- std across seeds) ===")
        summary = summarize(ma, ['Method'], os.path.join(analysis_dir, 'main_summary.csv'))
        print(summary.to_string(), "\n")
        for metric in [args.metric, 'F1-Score']:
            tests = paired_tests(ma, metric)
            if len(tests):
                print(f"--- Paired tests on {metric} ---")
                print(tests.to_string(index=False), "\n")
                tests.to_csv(os.path.join(analysis_dir, f'main_tests_{metric.replace(" ", "_")}.csv'), index=False)

    # --- robustness: method x ratio pivot + significance at each ratio ---
    rb = df[df['Experiment'] == 'robust']
    if len(rb):
        print("=== ROBUSTNESS (mean accuracy, method x poison ratio) ===")
        pivot = rb.pivot_table(index='Method', columns='Poison_Ratio',
                               values=args.metric, aggfunc='mean').round(2)
        print(pivot.to_string(), "\n")
        pivot.to_csv(os.path.join(analysis_dir, 'robust_pivot.csv'))
        pivot_std = rb.pivot_table(index='Method', columns='Poison_Ratio',
                                   values=args.metric, aggfunc='std').round(2)
        pivot_std.to_csv(os.path.join(analysis_dir, 'robust_pivot_std.csv'))
        tests = paired_tests(rb, args.metric, extra_group='Poison_Ratio')
        if len(tests):
            tests.to_csv(os.path.join(analysis_dir, 'robust_tests.csv'), index=False)

    # --- sensitivity ---
    sv = df[df['Experiment'] == 'sensitivity']
    if len(sv):
        print("=== SENSITIVITY (mean accuracy per variant x ratio) ===")
        pivot = sv.pivot_table(index='Variant', columns='Poison_Ratio',
                               values=args.metric, aggfunc=['mean', 'std']).round(2)
        print(pivot.to_string(), "\n")
        pivot.to_csv(os.path.join(analysis_dir, 'sensitivity_pivot.csv'))

    print(f"[INFO] Analysis CSVs written to {analysis_dir}/")

if __name__ == '__main__':
    main()
