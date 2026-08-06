"""Solo timing benchmark: measures per-round wall time for the three main methods
on an otherwise idle machine (the 'Training Time (s)' recorded during the parallel
revision runs was inflated by lane contention and must not be used in the paper).
10 measured rounds per method, extrapolated x10 for the 100-round figure.
Includes per-round test evaluation, matching how the original timings were taken.
"""
import json
import time

import src.preprocessing as prep
import src.fl_core as fl

DATA = "dataset/Train_Test_Windows_10.csv"
ROUNDS = 10

client_data, val_data, test_data, num_classes, input_dim, classes = \
    prep.load_and_process_data(DATA, num_clients=20, partition_seed=0)

results = {}
for method in ['fedavg', 'fedprox', 'fuzzy']:
    start = time.time()
    fl.run_fl_simulation(client_data, val_data, test_data, method, ROUNDS,
                         20, 5, 32, poison_ratio=0.0, seed=0, show_progress=False)
    dt = time.time() - start
    results[method] = {
        'rounds': ROUNDS,
        'total_s': round(dt, 2),
        'per_round_s': round(dt / ROUNDS, 3),
        'extrapolated_100r_s': round(dt * 10, 1),
    }
    print(f"[TIMING] {method}: {results[method]}", flush=True)

with open('results_revision/timing_solo.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print("[DONE] results_revision/timing_solo.json", flush=True)
