# SVM Credit Card Fraud Detection Lab

Interactive simulation web app for the project
**"Design of an SVM Model for Credit Card Fraud Detection and Classification"**.

## Features
- Synthetic transaction simulator (adjustable size and fraud rate) or upload of the Kaggle `creditcard.csv`
- SVM with selectable kernel (RBF, linear, polynomial, sigmoid), C and gamma
- Imbalance handling: none, class weights, SMOTE oversampling
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC and PR curves
- Experiments: imbalance comparison, kernel comparison, C x gamma grid-search heatmap
- SVM vs neural network (MLP) comparison
- PCA decision-boundary plot
- Live transaction-screening simulation

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `core.py` | Data generation, SMOTE, training, evaluation (no UI code) |
| `requirements.txt` | Python libraries |
| `.streamlit/config.toml` | Dark engineering theme |

## Run on your computer (VS Code)
```
python -m venv venv
venv\Scripts\activate            (Windows)
pip install -r requirements.txt
streamlit run app.py
```
Your browser opens at http://localhost:8501.

## Make it public so anyone can open it (free)
1. Create a free account at github.com and make a **new public repository**.
2. Upload all files from this folder (keep `.streamlit/config.toml` inside a `.streamlit` folder).
3. Go to **share.streamlit.io**, sign in with GitHub and click **Create app**.
4. Choose your repository, branch `main`, main file path `app.py`, then **Deploy**.
5. After a few minutes you get a public link like `https://your-name-svm-fraud.streamlit.app` that you can share with anyone.

Notes:
- Do not upload the large Kaggle CSV to GitHub (100 MB limit). Use the app's **Upload** option instead, or stay on the synthetic simulator.
- Free hosting has limited CPU. Keep "Legit training rows" at 12,000 or below for fast responses.
- Free apps go to sleep when unused; the first visit after a pause takes about a minute to wake.

## Suggested report structure
Introduction, problem statement, literature review, SVM theory (margin, kernel trick), methodology,
implementation, results and comparison tables, discussion (precision vs recall trade-off), conclusion, future scope.
