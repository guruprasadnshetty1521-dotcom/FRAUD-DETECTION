"""Core ML logic for the SVM credit card fraud detection project.

Kept separate from the Streamlit UI so it can be tested and reused.
"""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def make_synthetic(n_samples=40000, fraud_rate=0.01, n_features=20, seed=42):
    """Generate an imbalanced, credit-card-like dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[1 - fraud_rate, fraud_rate],
        class_sep=1.2,
        flip_y=0.002,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, n_features + 1)])
    df["Time"] = np.sort(rng.uniform(0, 172800, n_samples))
    amount = rng.lognormal(mean=3.5, sigma=1.0, size=n_samples)
    amount[y == 1] *= rng.uniform(1.0, 3.0, (y == 1).sum())
    df["Amount"] = amount
    df["Class"] = y
    return df


def prepare(df, test_size=0.2, train_cap=15000, seed=42):
    """Scale, split, and cap the number of legitimate training rows.

    SVM training time grows roughly with n^2, so we keep every fraud case
    and sample the legitimate class down to `train_cap` rows.
    """
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].astype(int)
    for col in ("Amount", "Time"):
        if col in X.columns:
            X[col] = StandardScaler().fit_transform(X[[col]])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    fraud_idx = y_tr[y_tr == 1].index
    legit_idx = y_tr[y_tr == 0].index
    if len(legit_idx) > train_cap:
        legit_idx = pd.Index(
            np.random.default_rng(seed).choice(legit_idx, train_cap, replace=False)
        )
    keep = fraud_idx.union(legit_idx)
    X_tr_small, y_tr_small = X_tr.loc[keep], y_tr.loc[keep]
    return X_tr_small, X_te, y_tr_small, y_te


def smote(X, y, k=5, seed=42):
    """Minimal SMOTE: synthesise minority samples between k-nearest neighbours."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_min = X[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())
    if len(X_min) < 2 or n_needed <= 0:
        return X, y
    k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(X_min), n_needed)
    pick = neigh[base, rng.integers(0, k, n_needed)]
    gap = rng.random((n_needed, 1))
    synth = X_min[base] + gap * (X_min[pick] - X_min[base])
    X_out = np.vstack([X, synth])
    y_out = np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])
    return X_out, y_out


def apply_imbalance(X, y, method):
    if method == "SMOTE oversampling":
        return smote(X, y)
    return np.asarray(X, dtype=float), np.asarray(y)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def train_svm(X, y, kernel="rbf", C=1.0, gamma="scale", imbalance="Class weights"):
    Xb, yb = apply_imbalance(X, y, imbalance)
    cw = "balanced" if imbalance == "Class weights" else None
    model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=cw)
    t = time.time()
    model.fit(Xb, yb)
    return model, time.time() - t


def train_ann(X, y, hidden=(64, 32), seed=42):
    Xb, yb = smote(X, y)
    model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300,
                          early_stopping=True, random_state=seed)
    t = time.time()
    model.fit(Xb, yb)
    return model, time.time() - t


def scores(model, X):
    """Continuous fraud score (higher = more fraudulent)."""
    X = np.asarray(X, dtype=float)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate(model, X_te, y_te):
    X_te = np.asarray(X_te, dtype=float)
    pred = model.predict(X_te)
    s = scores(model, X_te)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    return {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, s),
        "PR-AUC": average_precision_score(y_te, s),
        "cm": cm,
        "pred": pred,
        "scores": s,
    }


def kernel_comparison(X, y, X_te, y_te, C=1.0, imbalance="Class weights"):
    rows = []
    for k in ("linear", "poly", "rbf", "sigmoid"):
        m, t = train_svm(X, y, kernel=k, C=C, imbalance=imbalance)
        r = evaluate(m, X_te, y_te)
        rows.append({"Kernel": k, "Precision": r["Precision"], "Recall": r["Recall"],
                     "F1": r["F1"], "ROC-AUC": r["ROC-AUC"], "Train time (s)": t})
    return pd.DataFrame(rows)


def imbalance_comparison(X, y, X_te, y_te, kernel="rbf", C=1.0):
    rows = []
    for method in ("None (baseline)", "Class weights", "SMOTE oversampling"):
        m, t = train_svm(X, y, kernel=kernel, C=C, imbalance=method)
        r = evaluate(m, X_te, y_te)
        rows.append({"Method": method, "Accuracy": r["Accuracy"],
                     "Precision": r["Precision"], "Recall": r["Recall"],
                     "F1": r["F1"], "ROC-AUC": r["ROC-AUC"], "Train time (s)": t})
    return pd.DataFrame(rows)


def grid_search(X, y, Cs, gammas, kernel="rbf", sample=5000, seed=1):
    if len(X) > sample:
        idx = np.random.default_rng(seed).choice(len(X), sample, replace=False)
        Xg, yg = X.iloc[idx], y.iloc[idx]
        if yg.sum() < 10:  # make sure enough fraud rows are in the sample
            fr = np.where(y.values == 1)[0]
            idx = np.union1d(idx, fr)
            Xg, yg = X.iloc[idx], y.iloc[idx]
    else:
        Xg, yg = X, y
    gs = GridSearchCV(SVC(kernel=kernel, class_weight="balanced"),
                      {"C": Cs, "gamma": gammas}, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(np.asarray(Xg, dtype=float), np.asarray(yg))
    res = pd.DataFrame(gs.cv_results_)
    pivot = res.pivot(index="param_C", columns="param_gamma",
                      values="mean_test_score").astype(float)
    return gs.best_params_, gs.best_score_, pivot


def pca_boundary(X, y, kernel="rbf", C=1.0, gamma="scale", imbalance="Class weights",
                 max_points=6000, seed=0):
    """Fit an SVM on a 2-D PCA projection and return a decision grid."""
    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    pca = PCA(n_components=2, random_state=seed).fit(X)
    X2 = pca.transform(X)
    model, _ = train_svm(pd.DataFrame(X2), y.values, kernel, C, gamma, imbalance)
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, 250),
        np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, 250),
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return X2, np.asarray(y), xx, yy, Z, pca.explained_variance_ratio_
