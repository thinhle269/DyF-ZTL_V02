"""Full 100-round wall-clock benchmark, repeated, on an idle machine.
Addresses the internal-review request for actual (non-extrapolated) training
times with repetition. MUST be run alone (no concurrent training lanes),
otherwise contention inflates the numbers.

Output: results_revision/timing_full.json  {method: {runs: [...], mean_s, std_s}}
Resume-safe: completed (method, repeat) pairs are skipped.
"""
import json
import os
import time

import numpy as np

import src.preprocessing as prep
import src.fl_core as fl

DATA = "dataset/Train_Test_Windows_10.csv"
ROUNDS = 100
REPEATS = 3
OUT = "results_revision/timing_full.json"

results = {}
if os.path.exists(OUT):
    with open(OUT) as fh:
        results = json.load(fh)

client_data, val_data, test_data, num_classes, input_dim, classes = \
    prep.load_and_process_data(DATA, num_clients=20, partition_seed=0)

for method in ['fedavg', 'fedprox', 'fuzzy']:
    entry = results.setdefault(method, {'runs': []})
    while len(entry['runs']) < REPEATS:
        rep = len(entry['runs'])
        start = time.time()
        fl.run_fl_simulation(client_data, val_data, test_data, method, ROUNDS,
                             20, 5, 32, poison_ratio=0.0, seed=rep,
                             show_progress=False)
        dt = round(time.time() - start, 2)
        entry['runs'].append(dt)
        entry['mean_s'] = round(float(np.mean(entry['runs'])), 2)
        entry['std_s'] = round(float(np.std(entry['runs'], ddof=1)), 2) if len(entry['runs']) > 1 else None
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f"[TIMING-FULL] {method} repeat {rep}: {dt}s  (state saved)", flush=True)

print("[DONE]", json.dumps(results, indent=2), flush=True)
