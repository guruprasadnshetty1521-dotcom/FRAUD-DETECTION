"""SVM Credit Card Fraud Detection - interactive simulation dashboard.

Run locally:   streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

import core

# --------------------------------------------------------------------------
# Page + theme
# --------------------------------------------------------------------------
st.set_page_config(page_title="SVM Fraud Detection Lab", page_icon="🛡️", layout="wide")

BG, PANEL, GRID = "#0b1220", "#111a2e", "#24324d"
BLUE, CYAN, RED, GREEN, AMBER, TEXT = "#4f8cff", "#22d3ee", "#ff5c6c", "#34d399", "#fbbf24", "#e6edf7"

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID, "axes.grid": True, "grid.alpha": 0.5,
    "font.size": 10, "axes.titleweight": "bold",
})

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
.hero {{ background: linear-gradient(135deg, #0f1c3a 0%, #0b1220 60%);
        border: 1px solid {GRID}; border-left: 4px solid {BLUE};
        border-radius: 10px; padding: 18px 24px; margin-bottom: 14px; }}
.hero h1 {{ font-family: 'Courier New', monospace; font-size: 1.55rem; margin: 0; color: {TEXT}; }}
.hero p {{ margin: 4px 0 0 0; color: #93a4c3; font-size: 0.92rem; }}
.tag {{ display:inline-block; font-family:'Courier New',monospace; font-size:0.72rem;
        padding:2px 8px; margin-right:6px; border:1px solid {GRID}; border-radius:4px; color:{CYAN}; }}
.kpi {{ background:{PANEL}; border:1px solid {GRID}; border-radius:8px; padding:12px 14px; }}
.kpi .l {{ font-family:'Courier New',monospace; font-size:0.72rem; color:#93a4c3; letter-spacing:.06em; }}
.kpi .v {{ font-size:1.6rem; font-weight:700; color:{TEXT}; line-height:1.2; }}
.kpi .s {{ font-size:0.72rem; color:#6b7c9c; }}
.panel {{ background:{PANEL}; border:1px solid {GRID}; border-radius:8px; padding:14px 18px; }}
.feed-row {{ font-family:'Courier New',monospace; font-size:0.82rem; padding:3px 0; border-bottom:1px solid {GRID}; }}
h2, h3 {{ font-family:'Courier New',monospace; }}
</style>
<div class="hero">
  <span class="tag">SVM</span><span class="tag">ANN</span><span class="tag">SIMULATION</span>
  <h1>Design of an SVM Model for Credit Card Fraud Detection and Classification</h1>
  <p>Interactive simulation lab: tune the model, handle class imbalance, compare against a neural network, and watch live transaction screening.</p>
</div>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar: configuration
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
source = st.sidebar.radio("Data source", ["Synthetic simulation", "Upload creditcard.csv"])

if source == "Synthetic simulation":
    n_samples = st.sidebar.slider("Transactions", 10000, 100000, 40000, 5000)
    fraud_rate = st.sidebar.slider("Fraud rate (%)", 0.2, 5.0, 1.0, 0.1) / 100
    uploaded = None
else:
    uploaded = st.sidebar.file_uploader("CSV with a 'Class' column", type="csv")
    n_samples, fraud_rate = None, None

st.sidebar.markdown("---")
st.sidebar.subheader("SVM hyperparameters")
kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
C = st.sidebar.select_slider("C (regularisation)", [0.01, 0.1, 1, 10, 100], value=1)
gamma_opt = st.sidebar.select_slider("gamma", ["scale", 0.001, 0.01, 0.1, 1], value="scale")
imbalance = st.sidebar.selectbox("Imbalance handling",
                                 ["Class weights", "SMOTE oversampling", "None (baseline)"])
train_cap = st.sidebar.slider("Legit training rows (speed vs accuracy)", 2000, 30000, 12000, 1000)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
run = st.sidebar.button("▶ Train / Re-train model", type="primary", use_container_width=True)
st.sidebar.caption("Larger training sets give better models but SVM training time grows rapidly.")


# --------------------------------------------------------------------------
# Data + model pipeline (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating dataset…")
def get_data(n, rate, seed_):
    return core.make_synthetic(n, rate, seed=seed_)


@st.cache_data(show_spinner="Reading CSV…")
def read_csv(file):
    return pd.read_csv(file)


@st.cache_data(show_spinner="Preparing train / test split…")
def get_split(df, cap, seed_):
    return core.prepare(df, train_cap=cap, seed=seed_)


@st.cache_resource(show_spinner="Training SVM…")
def get_svm(X, y, kernel_, C_, gamma_, imb):
    return core.train_svm(X, y, kernel_, C_, gamma_, imb)


@st.cache_resource(show_spinner="Training neural network…")
def get_ann(X, y):
    return core.train_ann(X, y)


if source == "Synthetic simulation":
    df = get_data(n_samples, fraud_rate, int(seed))
elif uploaded is not None:
    df = read_csv(uploaded)
    if "Class" not in df.columns or df["Class"].nunique() < 2:
        st.error("The CSV must contain a 'Class' column with values 0 and 1.")
        st.stop()
else:
    st.info("Upload the Kaggle `creditcard.csv` in the sidebar, or switch to *Synthetic simulation*.")
    st.stop()

X_tr, X_te, y_tr, y_te = get_split(df, train_cap, int(seed))
gamma = gamma_opt if gamma_opt == "scale" else float(gamma_opt)

if run:  # force a fresh fit even if the parameters are unchanged
    get_svm.clear()
    get_ann.clear()
model, train_time = get_svm(X_tr, y_tr, kernel, float(C), gamma, imbalance)
res = core.evaluate(model, X_te, y_te)

tabs = st.tabs(["📘 Overview", "🗂️ Dataset", "📊 Results", "🧪 Experiments",
                "🧠 SVM vs ANN", "🗺️ Decision boundary", "📡 Live simulation"])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Problem statement")
        st.write(
            "Credit card fraud is rare (often under 0.2% of transactions) but costly. "
            "A classifier must catch as many fraudulent transactions as possible "
            "(high **recall**) without blocking too many genuine customers "
            "(high **precision**). Plain accuracy is misleading here: a model that "
            "labels everything 'legit' scores above 99% and detects no fraud at all."
        )
        st.subheader("Why a Support Vector Machine?")
        st.write(
            "An SVM finds the hyperplane that maximises the margin between the two classes. "
            "With the **kernel trick** (for example RBF) it separates classes that are not "
            "linearly separable by implicitly mapping features into a higher-dimensional space. "
            "Only the *support vectors* define the boundary, which suits small, hard-to-separate "
            "minority classes."
        )
        st.latex(r"\min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i \quad "
                 r"\text{s.t. } y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i,\ \xi_i\ge 0")
        st.latex(r"K(x,x') = \exp\!\left(-\gamma\,\|x-x'\|^2\right)")
    with c2:
        st.subheader("Methodology")
        st.markdown("""
<div class="panel">
<b>1.</b> Data acquisition (synthetic or Kaggle)<br>
<b>2.</b> Scaling of Time and Amount<br>
<b>3.</b> Stratified train / test split<br>
<b>4.</b> Imbalance handling (class weights, SMOTE)<br>
<b>5.</b> SVM training (kernel, C, gamma)<br>
<b>6.</b> Evaluation: precision, recall, F1, ROC-AUC, PR-AUC<br>
<b>7.</b> Comparison with an ANN (MLP)<br>
<b>8.</b> Live transaction screening simulation
</div>""", unsafe_allow_html=True)
        st.caption("Link to the ANN course: an RBF-kernel SVM behaves like an RBF neural network "
                   "whose hidden units are the support vectors.")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
with tabs[1]:
    n_fraud = int(df["Class"].sum())
    k = st.columns(4)
    kpi(k[0], "TRANSACTIONS", f"{len(df):,}")
    kpi(k[1], "FRAUD CASES", f"{n_fraud:,}")
    kpi(k[2], "FRAUD RATE", f"{100*n_fraud/len(df):.3f}%")
    kpi(k[3], "FEATURES", f"{df.shape[1]-1}")
    st.write("")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        counts = df["Class"].value_counts().sort_index()
        ax.bar(["Legit (0)", "Fraud (1)"], counts.values, color=[BLUE, RED])
        ax.set_yscale("log"); ax.set_title("Class distribution (log scale)")
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        amt = df["Amount"]
        bins = np.linspace(amt.min(), amt.quantile(0.99), 40)
        ax.hist(amt[df.Class == 0], bins=bins, alpha=0.7, color=BLUE, label="Legit", density=True)
        ax.hist(amt[df.Class == 1], bins=bins, alpha=0.7, color=RED, label="Fraud", density=True)
        ax.set_title("Transaction amount distribution"); ax.legend()
        st.pyplot(fig); plt.close(fig)
    st.subheader("Sample records")
    st.dataframe(df.head(50), use_container_width=True, height=260)
    st.caption(f"Training set after sampling: {len(X_tr):,} rows "
               f"({int(y_tr.sum()):,} fraud). Test set: {len(X_te):,} rows ({int(y_te.sum()):,} fraud).")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"SVM · kernel = {kernel} · C = {C} · gamma = {gamma} · {imbalance}")
    k = st.columns(6)
    kpi(k[0], "ACCURACY", f"{res['Accuracy']:.3f}", "misleading alone")
    kpi(k[1], "PRECISION", f"{res['Precision']:.3f}", "flagged that were fraud")
    kpi(k[2], "RECALL", f"{res['Recall']:.3f}", "fraud that was caught")
    kpi(k[3], "F1-SCORE", f"{res['F1']:.3f}")
    kpi(k[4], "ROC-AUC", f"{res['ROC-AUC']:.3f}")
    kpi(k[5], "TRAIN TIME", f"{train_time:.1f}s")
    st.write("")
    a, b, c = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        cm = res["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else TEXT, fontweight="bold")
        st.pyplot(fig); plt.close(fig)
    with b:
        fpr, tpr, _ = roc_curve(y_te, res["scores"])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(fpr, tpr, color=CYAN, lw=2, label=f"AUC = {res['ROC-AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#6b7c9c")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve"); ax.legend(loc="lower right")
        st.pyplot(fig); plt.close(fig)
    with c:
        p, r, _ = precision_recall_curve(y_te, res["scores"])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(r, p, color=AMBER, lw=2, label=f"AP = {res['PR-AUC']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve"); ax.legend(loc="upper right")
        st.pyplot(fig); plt.close(fig)
    tn, fp, fn, tp = res["cm"].ravel()
    st.info(f"Of {tp+fn:,} real frauds in the test set, the model caught **{tp:,}** and missed **{fn:,}**. "
            f"It also wrongly flagged **{fp:,}** genuine transactions.")

# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
with tabs[3]:
    st.write("Run controlled experiments. Each one retrains several SVMs, so it may take a short while.")
    e1, e2, e3 = st.tabs(["Imbalance methods", "Kernel comparison", "C × gamma grid search"])

    with e1:
        if st.button("Run imbalance comparison"):
            with st.spinner("Training 3 models…"):
                st.session_state["imb_df"] = core.imbalance_comparison(
                    X_tr, y_tr, X_te, y_te, kernel=kernel, C=float(C))
        if "imb_df" in st.session_state:
            d = st.session_state["imb_df"]
            st.dataframe(d.style.format({c: "{:.3f}" for c in d.columns if c != "Method"})
                         .highlight_max(subset=["Recall", "F1", "ROC-AUC"], color="#1f3b73"),
                         use_container_width=True)
            fig, ax = plt.subplots(figsize=(7, 3.4))
            x = np.arange(len(d)); w = 0.25
            ax.bar(x - w, d["Precision"], w, label="Precision", color=BLUE)
            ax.bar(x, d["Recall"], w, label="Recall", color=GREEN)
            ax.bar(x + w, d["F1"], w, label="F1", color=AMBER)
            ax.set_xticks(x, d["Method"]); ax.set_ylim(0, 1.05); ax.legend()
            ax.set_title("Effect of imbalance handling")
            st.pyplot(fig); plt.close(fig)
            st.caption("Class weights and SMOTE usually raise recall at the cost of precision. "
                       "Which is 'best' depends on the cost of a missed fraud versus a false alarm.")

    with e2:
        if st.button("Run kernel comparison"):
            with st.spinner("Training 4 models (linear can be slow)…"):
                st.session_state["ker_df"] = core.kernel_comparison(
                    X_tr, y_tr, X_te, y_te, C=float(C), imbalance=imbalance)
        if "ker_df" in st.session_state:
            d = st.session_state["ker_df"]
            st.dataframe(d.style.format({c: "{:.3f}" for c in d.columns if c != "Kernel"}),
                         use_container_width=True)
            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.bar(d["Kernel"], d["F1"], color=[BLUE, CYAN, GREEN, AMBER])
            ax.set_ylabel("F1-score"); ax.set_title("Kernel comparison")
            st.pyplot(fig); plt.close(fig)

    with e3:
        if st.button("Run grid search (3-fold CV)"):
            with st.spinner("Searching C × gamma…"):
                bp, bs, pv = core.grid_search(X_tr, y_tr, [0.1, 1, 10, 100],
                                              [0.001, 0.01, 0.1, 1], kernel=kernel if kernel != "linear" else "rbf")
                st.session_state["grid"] = (bp, bs, pv)
        if "grid" in st.session_state:
            bp, bs, pv = st.session_state["grid"]
            st.success(f"Best parameters: C = {bp['C']}, gamma = {bp['gamma']}  (CV F1 = {bs:.3f})")
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            im = ax.imshow(pv.values, cmap="viridis", aspect="auto"); ax.grid(False)
            ax.set_xticks(range(pv.shape[1]), [str(c) for c in pv.columns])
            ax.set_yticks(range(pv.shape[0]), [str(i) for i in pv.index])
            ax.set_xlabel("gamma"); ax.set_ylabel("C"); ax.set_title("Cross-validated F1")
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", color="white")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig); plt.close(fig)

# --------------------------------------------------------------------------
# SVM vs ANN
# --------------------------------------------------------------------------
with tabs[4]:
    st.write("The same data is used to train a Multi-Layer Perceptron (64 → 32 hidden neurons, "
             "trained on SMOTE-balanced data) for comparison with the SVM.")
    ann, ann_time = get_ann(X_tr, y_tr)
    ra = core.evaluate(ann, X_te, y_te)
    comp = pd.DataFrame({
        "Model": ["SVM", "ANN (MLP)"],
        "Precision": [res["Precision"], ra["Precision"]],
        "Recall": [res["Recall"], ra["Recall"]],
        "F1": [res["F1"], ra["F1"]],
        "ROC-AUC": [res["ROC-AUC"], ra["ROC-AUC"]],
        "Train time (s)": [train_time, ann_time],
    })
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if c != "Model"}),
                 use_container_width=True)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for name, r_, col in (("SVM", res, CYAN), ("ANN", ra, AMBER)):
            fpr, tpr, _ = roc_curve(y_te, r_["scores"])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {r_['ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#6b7c9c")
        ax.set_title("ROC: SVM vs ANN"); ax.legend(loc="lower right")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        m = ["Precision", "Recall", "F1"]
        x = np.arange(3); w = 0.35
        ax.bar(x - w / 2, [res[k] for k in m], w, label="SVM", color=CYAN)
        ax.bar(x + w / 2, [ra[k] for k in m], w, label="ANN", color=AMBER)
        ax.set_xticks(x, m); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Metric comparison")
        st.pyplot(fig); plt.close(fig)

# --------------------------------------------------------------------------
# Decision boundary
# --------------------------------------------------------------------------
with tabs[5]:
    st.write("The data is projected to 2-D with PCA and an SVM is fitted on the projection, "
             "so the decision boundary can be drawn. The full model above uses all features.")
    if st.button("Draw decision boundary"):
        with st.spinner("Fitting 2-D SVM…"):
            st.session_state["pca"] = core.pca_boundary(
                X_tr, y_tr, kernel=kernel, C=float(C), gamma=gamma, imbalance=imbalance)
    if "pca" in st.session_state:
        X2, y2, xx, yy, Z, ev = st.session_state["pca"]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 30), cmap="coolwarm", alpha=0.35)
        ax.contour(xx, yy, Z, levels=[0], colors="white", linewidths=1.8)
        ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=4, c=BLUE, alpha=0.5, label="Legit")
        ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=14, c=RED, edgecolors="white",
                   linewidths=0.3, label="Fraud")
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% variance)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% variance)")
        ax.set_title("SVM decision boundary (white line) on PCA projection"); ax.legend()
        st.pyplot(fig); plt.close(fig)

# --------------------------------------------------------------------------
# Live simulation
# --------------------------------------------------------------------------
with tabs[6]:
    st.write("Transactions arrive one by one from the held-out test set. The trained SVM screens each "
             "one in real time and flags suspected fraud.")
    c1, c2, c3 = st.columns(3)
    n_txn = c1.slider("Number of transactions", 10, 200, 40, 10)
    speed = c2.slider("Delay per transaction (s)", 0.0, 1.0, 0.15, 0.05)
    fraud_mix = c3.slider("Fraud share in stream (%)", 0, 50, 15, 5)

    if st.button("▶ Start live stream", type="primary"):
        rng = np.random.default_rng(int(time.time()) % 10000)
        fr_idx = np.where(y_te.values == 1)[0]
        lg_idx = np.where(y_te.values == 0)[0]
        n_fr = min(int(n_txn * fraud_mix / 100), len(fr_idx))
        idx = np.concatenate([rng.choice(fr_idx, n_fr, replace=False),
                              rng.choice(lg_idx, n_txn - n_fr, replace=False)])
        rng.shuffle(idx)

        Xv = X_te.values.astype(float)
        yv = y_te.values
        score_all = model.decision_function(Xv[idx])
        pred_all = model.predict(Xv[idx])

        m = st.columns(5)
        slots = [x.empty() for x in m]
        chart_slot = st.empty()
        feed_slot = st.empty()
        prog = st.progress(0.0)

        tp = fp = fn = tn = 0
        feed, hist = [], []
        for i in range(n_txn):
            p, t = int(pred_all[i]), int(yv[idx[i]])
            tp += p == 1 and t == 1; fp += p == 1 and t == 0
            fn += p == 0 and t == 1; tn += p == 0 and t == 0
            hist.append(score_all[i])
            amt = df.loc[X_te.index[idx[i]], "Amount"]
            flag = "🚨 FRAUD" if p else "✅ LEGIT"
            truth = "fraud" if t else "legit"
            ok = "" if p == t else "  ⚠ wrong"
            colr = RED if p else GREEN
            feed.insert(0, f'<div class="feed-row"><span style="color:{colr}">{flag}</span> '
                           f'· txn #{i+1:03d} · amount {amt:,.2f} · score {score_all[i]:+.2f} '
                           f'· actual: {truth}{ok}</div>')
            slots[0].metric("Screened", i + 1)
            slots[1].metric("Flagged", tp + fp)
            slots[2].metric("Frauds caught", tp)
            slots[3].metric("Frauds missed", fn)
            slots[4].metric("False alarms", fp)

            fig, ax = plt.subplots(figsize=(9, 2.2))
            cols = [RED if s > 0 else BLUE for s in hist]
            ax.bar(range(len(hist)), hist, color=cols, width=0.8)
            ax.axhline(0, color="white", lw=0.8)
            ax.set_xlim(-1, n_txn); ax.set_ylabel("SVM score")
            ax.set_title("Decision score per transaction (above 0 = flagged)")
            chart_slot.pyplot(fig); plt.close(fig)
            feed_slot.markdown("".join(feed[:8]), unsafe_allow_html=True)
            prog.progress((i + 1) / n_txn)
            time.sleep(speed)

        st.success(f"Stream finished: {tp} of {tp+fn} frauds caught, {fp} false alarms "
                   f"out of {tn+fp} genuine transactions.")

st.markdown("---")
st.caption("Built with scikit-learn and Streamlit · Artificial Neural Networks course project")
