# Resilience Rewired — Tangerine track, Sheridan Datathon 2026

Pick any customer, apply a shock in plain English, and get a plan ranked by what actually happened to
similar customers: every step shows the customer's own numbers, the evidence (rates, 95% CIs, p-values)
and what the bank can do.

## Run it
```bash
pip install -r requirements.txt
python site/web_app.py          # main app → http://127.0.0.1:8600
```
Backup front end (same engine): `streamlit run app.py`

`app_data/` is already built, so the app runs without the raw data. With the raw CSVs in `data/`,
the web app reads `monthly_summary.csv`, `loans.csv`, `dispositions.csv` and `clients.csv` directly, and
`python precompute.py` rebuilds `app_data/`.

## Optional: Gemini
Everything works without an AI key. To let Gemini read twist cards and rewrite the customer message, create
`.streamlit/secrets.toml` (never commit it):
```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
GEMINI_MODEL = "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.8-flash"   # tried in order
```
Gemini reads the card, then a rule-based reader checks it and fills in anything Gemini missed (shown on screen).
If Gemini is unavailable, the rule-based reader and a template message take over. Gemini never calculates numbers.

## Files
- `site/` — web app (design by our teammate; light and dark mode)
  - `web_app.py` — API + static server · `engine.py` — the judges' formulas (I, F, S, B, H, P, runway)
  - `web/` — the page (HTML, CSS, JS)
- `core.py` — risk model, evidence from similar customers, ranked plans, significance tests, the six findings
- `agent.py` — twist-card reader (Gemini + rule-based cross-check) and customer message
- `precompute.py` — builds `app_data/` from the raw CSVs (only step that reads transactions.csv)
- `discovery.py` — reproduces every finding from the raw data (output in `discovery_output.txt`)
- `check.py` — 52 automated checks: judges' example answers, twist-card phrasings, regression and statistical tests
- `app.py` — backup Streamlit front end on the same `core.py`

## Method
- Runway numbers follow the judges' definitions exactly (checked against the README examples).
- Evidence: features from each account's past; outcome = balance below zero in the next 12 months
  (3,660 accounts with 24+ months). Replicated on loan default using pre-loan data only (682 loans).
- Risk model: logistic regression on savings (months of spending), no transfers with other banks, past negative
  balance. Held-out AUC 0.91 (bootstrap 95% CI 0.84–0.96).
- Plans are ranked by the gap in predicted risk between the customer and customers already in the "after" state,
  and only when that gap's 95% CI (delta method) stays above zero. An association, not a measured causal effect.
- Not used: standing_orders and cards (end-of-data snapshot / mostly issued after loans → leakage).

## Data and tools
- PKDD'99 Discovery Challenge Financial dataset (CTU Prague): https://relational.fel.cvut.cz/dataset/Financial
- Built with Python, pandas, statsmodels, scikit-learn, SciPy, Gemini API; Streamlit and Plotly for the backup app.
  AI assistance (Claude) was used for coding and analysis.
