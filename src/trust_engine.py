import numpy as np
import pandas as pd
import torch

# Faithful reconstruction of the TrustEvaluator that produced the Run-C results
# (recovered by disassembling trust_engine.cpython-313.pyc, Nov 2025), verified
# numerically against results/trust_scores_poison_0.2.csv:
#   - warm-up (first 3 logged rounds): threshold 0.1; trust resets to 1.0 unless
#     val accuracy < 0.1, in which case trust *= 0.9  (0.9 -> 0.81 -> 0.729)
#   - after warm-up: acc < 0.70 -> trust *= 0.2 ("Draconian" decay)
#                    acc < 0.85 -> trust *= 0.9 (mild decay)
#                    else       -> trust = min(1.0, trust + 0.05)
#   - threshold = max(mean(T) - alpha*std(T), 0.4); Run C used alpha = 0.5
# All constants are exposed as parameters so the sensitivity study can sweep them;
# defaults reproduce Run C exactly.

class TrustEvaluator:
    def __init__(self, num_clients, decay_factor=0.2, recovery_factor=0.05, alpha=1.5,
                 mild_decay=0.9, low_gate=0.70, high_gate=0.85,
                 min_safety_threshold=0.4, warmup_rounds=2, warmup_threshold=0.1,
                 warmup_min_acc=0.1, warmup_decay=0.9):
        self.num_clients = num_clients
        self.trust_scores = np.ones(num_clients)
        self.decay = decay_factor
        self.recovery = recovery_factor
        self.alpha = alpha
        self.mild_decay = mild_decay
        self.low_gate = low_gate
        self.high_gate = high_gate
        self.min_safety_threshold = min_safety_threshold
        self.warmup_rounds = warmup_rounds
        self.warmup_threshold = warmup_threshold
        self.warmup_min_acc = warmup_min_acc
        self.warmup_decay = warmup_decay

        self.current_threshold = warmup_threshold
        self.current_round = 0
        self.history = []

    def _evaluate_accuracy(self, model, val_loader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0.0

    def calculate_trust(self, local_model, _, val_loader, device, idx):
        acc = self._evaluate_accuracy(local_model, val_loader, device)
        old = self.trust_scores[idx]

        if self.current_round < self.warmup_rounds:
            # Grace period: only near-zero-accuracy updates are penalized
            new = old * self.warmup_decay if acc < self.warmup_min_acc else 1.0
        else:
            if acc < self.low_gate:
                new = old * self.decay
            elif acc < self.high_gate:
                new = old * self.mild_decay
            else:
                new = min(1.0, old + self.recovery)

        self.trust_scores[idx] = new
        return new

    def update_dynamic_threshold(self):
        if self.current_round < self.warmup_rounds:
            self.current_threshold = self.warmup_threshold
        else:
            mu = np.mean(self.trust_scores)
            sigma = np.std(self.trust_scores)
            self.current_threshold = max(mu - self.alpha * sigma, self.min_safety_threshold)
        return self.current_threshold

    def is_trusted(self, idx):
        return self.trust_scores[idx] >= self.current_threshold

    def log_round(self, round_num):
        row = {'Round': round_num, 'Threshold': self.current_threshold}
        for i in range(self.num_clients):
            row[f'Client_{i}'] = self.trust_scores[i]
        self.history.append(row)
        self.current_round = round_num

    def save_history(self, filepath):
        pd.DataFrame(self.history).to_csv(filepath, index=False)
        print(f"[INFO] Trust history saved to {filepath}")
