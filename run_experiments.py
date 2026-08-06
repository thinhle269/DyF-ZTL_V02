"""
Reviewer-response experiment driver (Aug 2026 revision of DyF-ZTL).

Presets:
  smoke        quick pipeline check (2 rounds, all methods)         ~10 min
  main         multi-seed clean comparison, 100 rounds              (comment: statistical significance)
  ablation     FuzzyNoTrust / DeepTrust, multi-seed, 100 rounds     (comment: ablation study)
  robust       SOTA robust aggregators + all methods, poison sweep  (comment: SOTA comparison)
  sensitivity  trust-engine hyperparameter sweep at poison 0.2/0.4  (comment: sensitivity study)

Every run appends one row to <outdir>/runs.csv immediately (safe to interrupt).
Per-run artifacts (per-round metrics, confusion matrix, trust history) go to
<outdir>/<exp>/. Same seed => identical data partition and identical poisoned
client set for every method => paired statistical tests are valid.

Usage examples:
  python run_experiments.py --exp smoke
  python run_experiments.py --exp main --seeds 0 1 2 3 4 --rounds 100
  python run_experiments.py --exp robust --seeds 0 1 2 --rounds 20
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

try:
    import msvcrt
except ImportError:
    msvcrt = None

import src.preprocessing as prep
import src.fl_core as fl
import src.utils as utils

DATA_PATH = "dataset/Train_Test_Windows_10.csv"

# arch: client model | algo: local objective | defense: server-side aggregation rule
METHODS = {
    'FedAvg':       dict(arch='deep',  algo='fedavg',  defense=None),
    'FedProx':      dict(arch='deep',  algo='fedprox', defense=None),
    'DyF-ZTL':      dict(arch='fuzzy', algo='fedavg',  defense='trust'),
    'FuzzyNoTrust': dict(arch='fuzzy', algo='fedavg',  defense=None),      # ablation: fuzzy net, no trust engine
    'DeepTrust':    dict(arch='deep',  algo='fedavg',  defense='trust'),   # ablation: plain MLP + trust engine
    'Krum':         dict(arch='deep',  algo='fedavg',  defense='krum'),
    'MultiKrum':    dict(arch='deep',  algo='fedavg',  defense='multikrum'),
    'Median':       dict(arch='deep',  algo='fedavg',  defense='median'),
    'TrimmedMean':  dict(arch='deep',  algo='fedavg',  defense='trimmed'),
    'FLTrust':      dict(arch='deep',  algo='fedavg',  defense='fltrust'),
    'FLTrustFull':  dict(arch='deep',  algo='fedavg',  defense='fltrust',
                         sim=dict(fltrust_root_size=10**9)),  # root = entire D_val
    # DFNN + robust aggregation rules: decouples the architecture from the defense
    'FuzzyKrum':     dict(arch='fuzzy', algo='fedavg', defense='krum'),
    'FuzzyMultiKrum':dict(arch='fuzzy', algo='fedavg', defense='multikrum'),
    'FuzzyMedian':   dict(arch='fuzzy', algo='fedavg', defense='median'),
    'FuzzyTrimmed':  dict(arch='fuzzy', algo='fedavg', defense='trimmed'),
    'FuzzyFLTrust':  dict(arch='fuzzy', algo='fedavg', defense='fltrust'),
}

# One-at-a-time deviations from the faithful Run-C defaults
SENSITIVITY_GRID = [
    ('alpha', 0.25), ('alpha', 1.0), ('alpha', 1.5),
    ('decay_factor', 0.1), ('decay_factor', 0.4), ('decay_factor', 0.8),
    ('low_gate', 0.60), ('low_gate', 0.80),
    ('high_gate', 0.80), ('high_gate', 0.90),
    ('warmup_rounds', 0), ('warmup_rounds', 4),
    ('min_safety_threshold', 0.2), ('min_safety_threshold', 0.6),
    ('baseline', None),  # faithful defaults for reference
]

_DATA_CACHE = {}

def get_data(num_clients, seed):
    key = (num_clients, seed)
    if key not in _DATA_CACHE:
        _DATA_CACHE.clear()  # keep at most one partition in memory
        _DATA_CACHE[key] = prep.load_and_process_data(
            DATA_PATH, num_clients=num_clients, partition_seed=seed)
    return _DATA_CACHE[key]

def append_row(csv_path, row):
    # Lock-protected append so multiple parallel lanes can share one runs.csv.
    # analyze_results.py also unions the per-run final_*.json files, so even a
    # failed append cannot lose a result.
    df = pd.DataFrame([row])
    lock_path = csv_path + '.lock'
    for _ in range(50):
        lf = None
        try:
            lf = open(lock_path, 'a')
            if msvcrt is not None:
                try:
                    msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    lf.close()
                    time.sleep(0.2 + random.random() * 0.3)
                    continue
            try:
                header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
                df.to_csv(csv_path, mode='a', header=header, index=False)
                return
            finally:
                if msvcrt is not None:
                    try:
                        msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                lf.close()
        except Exception:
            if lf is not None:
                try: lf.close()
                except Exception: pass
            time.sleep(0.2)
    df.to_csv(f"{csv_path}.part{os.getpid()}.csv", mode='a', header=True, index=False)

def already_done(exp, method_label, seed, ratio, rounds, args):
    # Primary marker: the per-run final JSON (written atomically per run).
    tag = f"{method_label}_seed{seed}_p{ratio}"
    if os.path.exists(os.path.join(args.outdir, exp, f"final_{tag}.json")):
        return True
    runs_csv = os.path.join(args.outdir, "runs.csv")
    if not os.path.exists(runs_csv):
        return False
    try:
        done = pd.read_csv(runs_csv, on_bad_lines='skip')
        mask = ((done['Experiment'] == exp) & (done['Method'] == method_label) &
                (done['Seed'] == seed) & (done['Poison_Ratio'] == ratio) &
                (done['Rounds'] == rounds))
        return bool(mask.any())
    except Exception:
        return False

def run_one(exp, method, seed, ratio, rounds, args, trust_params=None, variant='', sim_kwargs=None):
    if already_done(exp, method + variant, seed, ratio, rounds, args):
        print(f"[SKIP] {method + variant} seed{seed} p{ratio} already done", flush=True)
        return None

    cfg = METHODS[method]
    tag = f"{method}{variant}_seed{seed}_p{ratio}"
    exp_dir = os.path.join(args.outdir, exp)
    os.makedirs(exp_dir, exist_ok=True)

    # Claim file: prevents two parallel lanes from starting the same run.
    claim = os.path.join(exp_dir, f"claim_{tag}")
    try:
        with open(claim, 'x') as fh:
            fh.write(str(os.getpid()))
    except FileExistsError:
        if time.time() - os.path.getmtime(claim) < 7200:
            print(f"[SKIP] {tag} claimed by another running process", flush=True)
            return None
        with open(claim, 'w') as fh:  # stale claim (>2h): take over
            fh.write(str(os.getpid()))

    client_data, val_data, test_data, num_classes, input_dim, label_classes = \
        get_data(args.clients, seed)

    trust_csv = os.path.join(exp_dir, f"trust_{tag}.csv") if cfg['defense'] == 'trust' else None

    print(f"\n=== [{exp}] {tag} (rounds={rounds}) ===", flush=True)
    start = time.time()
    extra = dict(cfg.get('sim', {}))
    extra.update(sim_kwargs or {})
    acc_hist, y_true, y_pred, full_metrics = fl.run_fl_simulation(
        client_data, val_data, test_data, 'fedavg', rounds,
        args.clients, args.epochs, args.batch,
        poison_ratio=ratio, seed=seed,
        arch=cfg['arch'], algo=cfg['algo'], defense=cfg['defense'],
        trust_params=trust_params, trust_csv_path=trust_csv,
        method_label=method + variant, show_progress=not args.quiet, **extra)
    duration = time.time() - start

    final_metrics, cm = utils.calculate_extended_metrics(y_true, y_pred, method + variant)

    pd.DataFrame(full_metrics).to_csv(os.path.join(exp_dir, f"per_round_{tag}.csv"), index=False)
    utils.save_confusion_matrix_csv(cm, label_classes, tag, out_dir=exp_dir)

    row = dict(final_metrics)
    row.update({'Experiment': exp, 'Seed': seed, 'Poison_Ratio': ratio,
                'Rounds': rounds, 'Variant': variant.lstrip('_'),
                'Training Time (s)': round(duration, 2)})

    with open(os.path.join(exp_dir, f"final_{tag}.json"), 'w') as fh:
        json.dump({'metrics': row, 'acc_history': acc_hist}, fh, indent=2)
    append_row(os.path.join(args.outdir, "runs.csv"), row)
    try:
        os.remove(claim)
    except OSError:
        pass

    print(f"[DONE] {tag}: Acc={final_metrics['Accuracy']} F1={final_metrics['F1-Score']} "
          f"FAR={final_metrics['False Alarm Rate (FAR)']} ({duration:.0f}s)", flush=True)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    choices=['smoke', 'main', 'ablation', 'robust', 'sensitivity',
                             'ablation_poison', 'valsize', 'fuzzyagg'])
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--ratios', type=float, nargs='+', default=None)
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--clients', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--outdir', default='results_revision')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.exp == 'smoke':
        seeds = args.seeds or [0]
        rounds = args.rounds or 2
        for m in ['FedAvg', 'FedProx', 'DyF-ZTL', 'FuzzyNoTrust', 'DeepTrust']:
            run_one('smoke', m, seeds[0], 0.0, rounds, args)
        for m in ['Krum', 'MultiKrum', 'Median', 'TrimmedMean', 'FLTrust', 'DyF-ZTL']:
            run_one('smoke', m, seeds[0], 0.4, rounds, args)

    elif args.exp == 'main':
        seeds = args.seeds or [0, 1, 2, 3, 4]
        rounds = args.rounds or 100
        methods = args.methods or ['FedAvg', 'FedProx', 'DyF-ZTL']
        for seed in seeds:
            for m in methods:
                run_one('main', m, seed, 0.0, rounds, args)

    elif args.exp == 'ablation':
        seeds = args.seeds or [0, 1, 2, 3, 4]
        rounds = args.rounds or 100
        methods = args.methods or ['FuzzyNoTrust', 'DeepTrust']
        for seed in seeds:
            for m in methods:
                run_one('ablation', m, seed, 0.0, rounds, args)

    elif args.exp == 'robust':
        seeds = args.seeds or [0, 1, 2]
        rounds = args.rounds or 20
        ratios = args.ratios or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        methods = args.methods or ['FedAvg', 'FedProx', 'DyF-ZTL', 'Krum', 'MultiKrum',
                                   'Median', 'TrimmedMean', 'FLTrust',
                                   'FuzzyNoTrust', 'DeepTrust']
        for seed in seeds:
            for ratio in ratios:
                for m in methods:
                    run_one('robust', m, seed, ratio, rounds, args)

    elif args.exp == 'ablation_poison':
        # Reviewer request: ablation under attack (DyF-ZTL and FedAvg rows come from 'robust')
        seeds = args.seeds or [0, 1, 2]
        rounds = args.rounds or 20
        ratios = args.ratios or [0.4, 0.5]
        methods = args.methods or ['FuzzyNoTrust', 'DeepTrust']
        for seed in seeds:
            for ratio in ratios:
                for m in methods:
                    run_one('ablation_poison', m, seed, ratio, rounds, args)

    elif args.exp == 'valsize':
        # Reviewer request: trusted-set-size sensitivity + FLTrust with full D_val
        seeds = args.seeds or [0, 1, 2]
        rounds = args.rounds or 20
        ratios = args.ratios or [0.4]
        for seed in seeds:
            for ratio in ratios:
                for size in [100, 300, 600, 1200]:
                    run_one('valsize', 'DyF-ZTL', seed, ratio, rounds, args,
                            sim_kwargs={'trust_val_size': size}, variant=f'_val{size}')
                run_one('valsize', 'FLTrustFull', seed, ratio, rounds, args)

    elif args.exp == 'fuzzyagg':
        # Reviewer request: DFNN combined with each robust aggregation rule
        seeds = args.seeds or [0, 1, 2]
        rounds = args.rounds or 20
        ratios = args.ratios or [0.2, 0.4, 0.6, 0.8]
        methods = args.methods or ['FuzzyKrum', 'FuzzyMultiKrum', 'FuzzyMedian',
                                   'FuzzyTrimmed', 'FuzzyFLTrust']
        for seed in seeds:
            for ratio in ratios:
                for m in methods:
                    run_one('fuzzyagg', m, seed, ratio, rounds, args)

    elif args.exp == 'sensitivity':
        seeds = args.seeds or [0, 1, 2]
        rounds = args.rounds or 20
        ratios = args.ratios or [0.2, 0.4]
        for seed in seeds:
            for ratio in ratios:
                for param, value in SENSITIVITY_GRID:
                    if param == 'baseline':
                        tp, variant = None, '_baseline'
                    else:
                        tp, variant = {param: value}, f"_{param}={value}"
                    run_one('sensitivity', 'DyF-ZTL', seed, ratio, rounds, args,
                            trust_params=tp, variant=variant)

    print("\n[ALL DONE]", flush=True)

if __name__ == '__main__':
    main()
