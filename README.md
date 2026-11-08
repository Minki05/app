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
Gemini reads the card, then a rule-based reader checks it and fills in anything Gemini missed. The reading is shown
field by field (green = stated in the card, amber = assumed or worked out, with the reason; red = not turned into a
number) and nothing is calculated until the person edits or confirms it ("Confirm and calculate").
If Gemini is unavailable, the rule-based reader and a template message take over. Gemini never calculates numbers.

## Files
- `site/` — web app (design by our teammate; light and dark mode)
  - `web_app.py` — API + static server · `engine.py` — the judges' formulas (I, F, S, B, H, P, runway)
  - `web/` — the page (HTML, CSS, JS)
- `core.py` — risk model, evidence from similar customers, ranked plans, significance tests, the six findings
- `agent.py` — twist-card reader (Gemini + rule-based cross-check) and customer message
- `precompute.py` — builds `app_data/` from the raw CSVs (only step that reads transactions.csv)
- `discovery.py` — reproduces every finding from the raw data (output in `discovery_output.txt`)
- `check.py` — 61 automated checks: judges' example answers, twist-card phrasings, regression and statistical tests
- `site/check_web.py` — 128 checks through the real web server and page
- `site/check_guardrails.py` — 112 demo guardrails, run with AI switched off, including 250 random accounts x
  random scenarios (with and without timing) where every output must reach the same conclusion: verdict, shortfall,
  monthly change needed, plan result, guide, message and chart. `RR_N=1500 RR_SEED=7` runs more.
- `app.py` — backup Streamlit front end on the same `core.py`

## Method
- Runway numbers follow the judges' definitions exactly (checked against the README examples).
- A shock with timing (starts later, ends after N months) is answered by one month-by-month engine for every figure
  on screen; with no timing that engine and the judges' runway agree, so the judges' number is used as is.
- Plans are labelled by when they apply: "now, on today's income" (e.g. building the emergency fund) or "if the
  shock hits, on the income after it".
- Evidence: features from each account's past; outcome = balance below zero in the next 12 months
  (3,660 accounts with 24+ months). Replicated on loan default using pre-loan data only (682 loans).
- Risk model: logistic regression on savings (months of spending), no transfers with other banks, past negative
  balance. AUC 0.91 on a 30% hold-out of accounts (bootstrap 95% CI 0.84–0.96): how well it ranks accounts that went
  below $0 above those that didn't. It is not a percent-correct.
- Plans are ranked by the gap in predicted risk between the customer and customers already in the "after" state,
  and only when that gap's 95% CI (delta method) stays above zero. An association, not a measured causal effect.
- Savings goals (emergency fund, refilling a repair, keeping 4 loan payments) are money still to be saved: they are
  never added to today's cash, the runway or the chart. Only a confirmed monthly cut or extra money in changes the
  plan line, and it starts from the same cash. One-off money in (`windfall`) is a shock input and is added as before.
- Not used: standing_orders and cards (end-of-data snapshot / mostly issued after loans → leakage).

## Data and tools
- PKDD'99 Discovery Challenge Financial dataset (CTU Prague): https://relational.fel.cvut.cz/dataset/Financial
- Built with Python, pandas, statsmodels, scikit-learn, SciPy, Gemini API; Streamlit and Plotly for the backup app.
  AI assistance (Claude) was used for coding and analysis.
