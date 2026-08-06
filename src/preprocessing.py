import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import torch

# Faithful restoration of the pipeline that produced the paper results (Run C,
# Nov 2025; recovered from src/fix_index_error.py == preprocessing.cpython-313.pyc),
# with one additive change: `partition_seed` makes the Dirichlet partition
# reproducible. partition_seed=None reproduces the legacy unseeded behavior.

def load_and_process_data(filepath, num_clients=10, non_iid_alpha=0.5, partition_seed=None):
    print(f"[INFO] Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] File not found. Please put 'Train_Test_Windows_10.csv' in 'dataset/' folder.")
        exit()

    # 1. DATA CLEANING
    print("[INFO] Cleaning data (removing spaces and non-numeric values)...")

    df.columns = df.columns.str.strip()

    if 'type' in df.columns:
        df = df[df['type'] != 'mitm']

    if 'type' not in df.columns:
        print("[ERROR] Column 'type' not found in dataset!")
        exit()

    y_raw = df['type']
    X_raw = df.drop(columns=['label', 'type', 'ts', 'date', 'time'], errors='ignore')

    # Force Numeric
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce')
    X_raw.dropna(axis=1, how='all', inplace=True)

    valid_indices = X_raw.dropna().index
    n_dirty = len(X_raw) - len(valid_indices)
    if n_dirty > 0:
        print(f"[WARN] Dropping {n_dirty} dirty rows containing non-numeric values...")

    X_raw = X_raw.loc[valid_indices]
    y_raw = y_raw.loc[valid_indices]

    X_raw.reset_index(drop=True, inplace=True)
    y_raw.reset_index(drop=True, inplace=True)

    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    print(f"[INFO] Classes detected: {classes}")

    # 2. STRICT SPLITTING STRATEGY (70/15/15)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, train_size=0.7, stratify=y, random_state=42
    )

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"[INFO] Data Splits: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # 3. FEATURE SELECTION (Fit on TRAIN only)
    print("[INFO] Performing Feature Selection (RF-based on Train data)...")

    sample_size = min(10000, len(y_train))
    X_sub, _, y_sub, _ = train_test_split(X_train_raw, y_train, train_size=sample_size, stratify=y_train, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_sub, y_sub)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = 15
    top_features = X_raw.columns[indices[:top_k]]
    print(f"[INFO] Top {top_k} Features selected: {list(top_features)}")

    X_train = X_train_raw[top_features].values
    X_val = X_val_raw[top_features].values
    X_test = X_test_raw[top_features].values

    # 4. SCALING
    print("[INFO] Applying StandardScaler (Fit on D_train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5. PARTITIONING D_TRAIN
    print(f"[INFO] Partitioning D_train for {num_clients} clients (Non-IID alpha={non_iid_alpha}, seed={partition_seed})...")

    rng = np.random.RandomState(partition_seed) if partition_seed is not None else np.random

    client_data = []
    min_size = 0
    N = len(y_train)

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(len(classes)):
            idx_k = np.where(y_train == k)[0]
            if len(idx_k) == 0: continue

            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(non_iid_alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for i in range(num_clients):
        X_c = X_train[idx_batch[i]]
        y_c = y_train[idx_batch[i]]

        if len(np.unique(y_c)) > 1 and len(y_c) > 30:
            try:
                smote = SMOTE(k_neighbors=1, random_state=42)
                X_c, y_c = smote.fit_resample(X_c, y_c)
            except:
                pass

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_c, dtype=torch.float32),
            torch.tensor(y_c, dtype=torch.long)
        )
        client_data.append(dataset)

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    return client_data, val_dataset, test_dataset, len(classes), len(top_features), classes
