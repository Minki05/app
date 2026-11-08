# Resilience Engine — Tangerine track, Sheridan Datathon 2026

Pick any customer, apply a shock, get a personal plan, and see **why** each step is recommended:
the customer's own numbers plus what actually happened to similar customers in the data.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
`app_data/` is already built. To rebuild it from the raw data (put the CSVs in `data/`):
```bash
python precompute.py
```

## Optional: Gemini
The app works without any AI key. To let Gemini read twist cards and write customer messages, create
`.streamlit/secrets.toml` (never commit it):
```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
GEMINI_MODEL = "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.8-flash"   # tried in order
```
If Gemini is missing or fails, a rule-based reader and a template take over. Gemini never calculates numbers.

## Files
- `app.py` — Streamlit app (know the customer → shock → plan with reasons; Advisor and Customer views)
- `agent.py` — reads twist cards in plain English and writes the customer message (Gemini with rule-based fallback)
- `precompute.py` — builds `app_data/` from the raw CSVs (only step that reads transactions.csv)
- `engine.py` — the same engine as a notebook script (prints plans)
- `discovery.py` — reproduces every finding from the raw data (feature screen with FDR, main finding, loan-default replication, leakage check, what doesn't matter)
- `validation.py` — evidence behind every recommendation (rates, 95% CIs, p-values, risk model, AUC)

## Method
- Engine numbers (I, F, S, B, H, runway) follow the judges' definitions exactly.
- Evidence: features from each account's past; outcome = balance below zero in the next 12 months
  (3,660 accounts with 24+ months). Replicated on loan default using pre-loan data only (682 loans).
- Risk model: logistic regression on buffer (months of spending), no-other-bank-transfer account, past negative balance.
  AUC 0.92 on held-out accounts. Correlation, not causation.
- Not used: standing_orders and cards (end-of-data snapshot / issued after loans → leakage).

## Data and tools
- PKDD'99 Discovery Challenge Financial dataset (CTU Prague): https://relational.fel.cvut.cz/dataset/Financial
- Built with Python, pandas, statsmodels, scikit-learn, Streamlit, Plotly. AI assistance (Claude) was used for coding and analysis.


## Web version (site/)
A second front end built with our teammate's design (light and dark mode). Same numbers: runway from
`site/engine.py` (the judging formulas), plans, risk, findings from `core.py`, twist reader and messages from `agent.py`.

1. Put the raw CSVs `monthly_summary.csv`, `loans.csv`, `dispositions.csv`, `clients.csv` in `data/` (or `site/data/`).
2. Run:
```bash
python site/web_app.py      # → http://127.0.0.1:8600
```
The Gemini key is read from `.streamlit/secrets.toml`, the same file the Streamlit app uses.
