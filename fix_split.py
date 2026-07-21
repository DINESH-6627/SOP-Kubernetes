

import os, sys, pickle, numpy as np
from collections import defaultdict

DATA_DIR = os.path.expanduser(
    "~/Downloads/kubernetes/data"
)

# Load sequences built in stage3
X = np.concatenate([
    np.load(os.path.join(DATA_DIR, "X_train.npy")),
    np.load(os.path.join(DATA_DIR, "X_val.npy")),
    np.load(os.path.join(DATA_DIR, "X_test.npy")),
])
y = np.concatenate([
    np.load(os.path.join(DATA_DIR, "y_train.npy")),
    np.load(os.path.join(DATA_DIR, "y_val.npy")),
    np.load(os.path.join(DATA_DIR, "y_test.npy")),
])

print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}")

rng = np.random.RandomState(42)

class_indices = defaultdict(list)
for idx, label in enumerate(y):
    class_indices[int(label)].append(idx)

train_idx, val_idx, test_idx = [], [], []

print(f"\n{'Class':<22} {'Total':>6} {'Train':>6} "
      f"{'Val':>5} {'Test':>5}  Note")
print("-" * 65)

for class_label in sorted(class_indices.keys()):
    indices  = class_indices[class_label]
    n        = len(indices)
    shuffled = rng.permutation(indices).tolist()

    # Guarantee threshold — any class with <=10 sequences
    # gets at least 1 in val and 1 in test
    GUARANTEE_THRESHOLD = 10

    if n <= GUARANTEE_THRESHOLD:
        # Reserve 1 for val, 1 for test, rest for train
        val_idx.append(shuffled[0])
        test_idx.append(shuffled[1])
        train_idx.extend(shuffled[2:])
        note = f"guaranteed (n={n})"
    else:
        # Normal stratified split
        n_val   = max(1, int(round(n * 0.15)))
        n_test  = max(1, int(round(n * 0.15)))
        n_train = n - n_val - n_test
        val_idx.extend(shuffled[:n_val])
        test_idx.extend(shuffled[n_val:n_val + n_test])
        train_idx.extend(shuffled[n_val + n_test:])
        note = "stratified"

    n_tr = len([i for i in train_idx
                if int(y[i]) == class_label])
    n_vl = len([i for i in val_idx
                if int(y[i]) == class_label])
    n_ts = len([i for i in test_idx
                if int(y[i]) == class_label])

    from config import CLASS_NAMES
    name = CLASS_NAMES.get(class_label, str(class_label))
    print(f"{name:<22} {n:>6} {n_tr:>6} {n_vl:>5} "
          f"{n_ts:>5}  {note}")

# Shuffle train
perm    = rng.permutation(len(train_idx))
tr_arr  = np.array(train_idx)[perm]

X_train = X[tr_arr]
y_train = y[tr_arr]
X_val   = X[val_idx]
y_val   = y[val_idx]
X_test  = X[test_idx]
y_test  = y[test_idx]

print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  "
      f"Test: {X_test.shape}")

# Verify all classes present
print("\nClasses in train:", sorted(set(y_train.tolist())))
print("Classes in val  :", sorted(set(y_val.tolist())))
print("Classes in test :", sorted(set(y_test.tolist())))

missing_val  = set(range(11)) - set(y_val.tolist())
missing_test = set(range(11)) - set(y_test.tolist())
print("Missing from val :", missing_val  or "none ✓")
print("Missing from test:", missing_test or "none ✓")

if not missing_val and not missing_test:
    # Re-fit scaler on train only
    from sklearn.preprocessing import StandardScaler
    D            = X_train.shape[2]
    scaler       = StandardScaler()
    X_train_flat = X_train.reshape(-1, D)
    X_val_flat   = X_val.reshape(-1, D)
    X_test_flat  = X_test.reshape(-1, D)

    X_train = scaler.fit_transform(
        X_train_flat).reshape(X_train.shape).astype(np.float32)
    X_val   = scaler.transform(
        X_val_flat).reshape(X_val.shape).astype(np.float32)
    X_test  = scaler.transform(
        X_test_flat).reshape(X_test.shape).astype(np.float32)

    # Save
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

    with open(os.path.join(DATA_DIR, "scaler_train.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n✓ All splits saved with guaranteed class coverage.")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")
else:
    print("\n✗ Still missing classes — check output above.")