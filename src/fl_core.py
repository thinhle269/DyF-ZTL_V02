import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.models
from src.trust_engine import TrustEvaluator
import src.utils as utils
import src.aggregators as aggs

# v4 = faithful restoration of the v3 fl_core that produced the Run-C results
# (update_project_v3.py + fl_core.cpython-313.pyc), extended additively for the
# reviewer-response experiments:
#   - seed: full reproducibility (torch init, DataLoader shuffling, poisoned-client
#     selection). Poisoned sets depend only on (seed, poison_ratio), so every
#     method at the same seed faces the identical attack -> paired statistics.
#   - arch/algo/defense decoupled for ablations (e.g. DeepNet+Trust, FuzzyNet alone).
#   - defenses: 'trust' (the proposed engine), 'krum', 'multikrum', 'median',
#     'trimmed', 'fltrust' (SOTA robust-aggregation baselines), or None.
#   - trust_params: dict overriding TrustEvaluator constants (sensitivity study).
#   - FedProx proximal term is now the canonical (mu/2)*||w - w_global||^2
#     (the Nov-2025 code summed unsquared layer norms).
# Calling with model_type in {'fedavg','fedprox','fuzzy'} and defaults reproduces
# the legacy behavior (modulo the FedProx fix and seeding).

LEGACY_CONFIGS = {
    'fedavg':  dict(arch='deep',  algo='fedavg',  defense=None),
    'fedprox': dict(arch='deep',  algo='fedprox', defense=None),
    'fuzzy':   dict(arch='fuzzy', algo='fedavg',  defense='trust'),
}

FLTRUST_ROOT_SIZE = 300

class LocalUpdate:
    def __init__(self, dataset, batch_size, learning_rate, epochs, device, algo='fedavg', mu=0.01):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device
        self.algo = algo
        self.mu = mu

    def train(self, model, global_model=None):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_loss = []
        for _ in range(self.epochs):
            batch_loss = []
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if self.algo == 'fedprox' and global_model is not None:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += ((w - w_t) ** 2).sum()
                    loss = loss + (self.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def average_weights(w):
    return aggs.average_weights(w)

def evaluate_model(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred

def _build_model(arch, input_dim, num_classes, device):
    if arch == 'fuzzy':
        return src.models.DynamicFuzzyNet(input_dim, num_classes).to(device)
    return src.models.DeepNet(input_dim, num_classes).to(device)

def _stratified_root_subset(dataset, size, seed):
    labels = np.array([int(y) for _, y in dataset])
    rng = np.random.RandomState(seed)
    picked = []
    classes, counts = np.unique(labels, return_counts=True)
    for c, cnt in zip(classes, counts):
        idx_c = np.where(labels == c)[0]
        n_c = max(1, int(round(size * cnt / len(labels))))
        picked.extend(rng.choice(idx_c, min(n_c, len(idx_c)), replace=False).tolist())
    return torch.utils.data.Subset(dataset, picked)

def run_fl_simulation(client_datasets, val_dataset, test_dataset, model_type, rounds,
                      num_clients, epochs, batch_size, poison_ratio=0.0,
                      seed=None, arch=None, algo=None, defense='auto',
                      trust_params=None, lr=0.01, mu=0.01,
                      trust_csv_path=None, method_label=None, show_progress=True,
                      trust_val_size=None, fltrust_root_size=FLTRUST_ROOT_SIZE):
    cfg = LEGACY_CONFIGS.get(model_type, LEGACY_CONFIGS['fedavg'])
    arch = arch or cfg['arch']
    algo = algo or cfg['algo']
    if defense == 'auto':
        defense = cfg['defense']
    method_label = method_label or model_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    input_dim = client_datasets[0][0][0].shape[0]
    all_labels = [int(y) for _, y in test_dataset]
    num_classes = len(set(all_labels))

    global_model = _build_model(arch, input_dim, num_classes, device)
    global_weights = global_model.state_dict()

    acc_history = []
    full_metrics_history = []
    final_y_true, final_y_pred = [], []

    trust_engine = None
    val_loader = None
    if defense == 'trust':
        tp = dict(alpha=0.5)  # Run-C value (v3 fl_core passed alpha=0.5 explicitly)
        if trust_params:
            tp.update(trust_params)
        trust_engine = TrustEvaluator(num_clients, **tp)
        trust_val_ds = val_dataset
        if trust_val_size is not None and trust_val_size < len(val_dataset):
            trust_val_ds = _stratified_root_subset(val_dataset, trust_val_size, seed if seed is not None else 0)
        val_loader = DataLoader(trust_val_ds, batch_size=256, shuffle=False)

    root_dataset = None
    if defense == 'fltrust':
        if fltrust_root_size >= len(val_dataset):
            root_dataset = val_dataset
        else:
            root_dataset = _stratified_root_subset(val_dataset, fltrust_root_size, seed if seed is not None else 0)

    num_poisoned = int(num_clients * poison_ratio)
    poison_rng = np.random.RandomState(seed) if seed is not None else np.random
    poisoned_indices = poison_rng.choice(range(num_clients), num_poisoned, replace=False)
    f_assumed = num_poisoned  # oracle knowledge granted to Krum/Trimmed-Mean baselines
    if num_poisoned > 0:
        print(f"[WARN] Poisoning clients: {sorted(poisoned_indices.tolist())}")

    iterator = range(rounds)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Training {method_label} (Poison={poison_ratio})")

    for r in iterator:
        local_weights_candidates = []
        client_indices_candidates = []

        for idx in range(num_clients):
            local_model = _build_model(arch, input_dim, num_classes, device)
            local_model.load_state_dict(global_weights)

            dataset_to_use = client_datasets[idx]
            if idx in poisoned_indices:
                X_p, y_p = dataset_to_use[:][0].clone(), dataset_to_use[:][1].clone()
                y_p = (y_p + 1) % num_classes
                dataset_to_use = torch.utils.data.TensorDataset(X_p, y_p)

            trainer = LocalUpdate(dataset_to_use, batch_size, lr, epochs, device, algo=algo, mu=mu)
            w, _ = trainer.train(local_model, global_model if algo == 'fedprox' else None)

            if defense == 'trust':
                local_model.load_state_dict(w)
                trust_engine.calculate_trust(local_model, None, val_loader, device, idx)

            local_weights_candidates.append(w)
            client_indices_candidates.append(idx)

        # --- AGGREGATION / DEFENSE ---
        if defense == 'trust':
            trust_engine.update_dynamic_threshold()
            final_local_weights = [w for i, w in zip(client_indices_candidates, local_weights_candidates)
                                   if trust_engine.is_trusted(i)]
            trust_engine.log_round(r)
            if len(final_local_weights) > 0:
                global_weights = aggs.average_weights(final_local_weights)
        elif defense == 'krum':
            global_weights, _ = aggs.krum(local_weights_candidates, f_assumed, multi=False)
        elif defense == 'multikrum':
            global_weights, _ = aggs.krum(local_weights_candidates, f_assumed, multi=True)
        elif defense == 'median':
            global_weights = aggs.coordinate_median(local_weights_candidates)
        elif defense == 'trimmed':
            global_weights = aggs.trimmed_mean(local_weights_candidates, f_assumed)
        elif defense == 'fltrust':
            server_model = _build_model(arch, input_dim, num_classes, device)
            server_model.load_state_dict(global_weights)
            server_trainer = LocalUpdate(root_dataset, batch_size, lr, epochs, device, algo='fedavg')
            root_w, _ = server_trainer.train(server_model)
            global_weights, _ = aggs.fltrust(global_weights, local_weights_candidates, root_w)
        else:
            global_weights = aggs.average_weights(local_weights_candidates)

        global_model.load_state_dict(global_weights)

        # --- EVALUATION & LOGGING PER ROUND ---
        y_true, y_pred = evaluate_model(global_model, test_dataset, device)
        round_metrics, _ = utils.calculate_extended_metrics(y_true, y_pred, method_label)
        round_metrics['Round'] = r
        full_metrics_history.append(round_metrics)
        acc_history.append(round_metrics['Accuracy'])

        if r == rounds - 1:
            final_y_true = y_true
            final_y_pred = y_pred

    if trust_engine is not None and trust_csv_path:
        trust_engine.save_history(trust_csv_path)

    return acc_history, final_y_true, final_y_pred, full_metrics_history
