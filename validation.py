# =====================================================================
# Evidence & validation (run AFTER engine.py in the same notebook)
# Shows where every number in the recommendations comes from.
# =====================================================================
from scipy.stats import fisher_exact
pd.set_option("display.width", 160)

def rate_table(groups, outcome, title):
    t = outcome.groupby(groups, observed=True).agg(events="sum", n="count")
    lo, hi = proportion_confint(t.events, t.n, method="wilson")
    t["rate"] = (t.events / t.n).map("{:.1%}".format)
    t["95% CI"] = [f"{a:.1%} - {b:.1%}" for a, b in zip(lo, hi)]
    print(f"\n--- {title} ---"); print(t)

print("=" * 70)
print("SETUP: features from each account's past; outcome = went negative in the NEXT 12 months")
print(f"Accounts: {len(train)} | Outcome rate: {y.mean():.1%}")
print("=" * 70)

# 1. Does the judges' runway predict the future? (validates the engine)
rate_table(pd.cut(train.job_loss_runway, [-1, 3, 6, 12, 121],
                  labels=["<3 mo", "3-6 mo", "6-12 mo", "12+ mo"], right=False), y,
           "1. Job-loss runway (judge formula) vs going negative next year")

# 2. Buffer in months of spending (used for the emergency-fund plan)
rate_table(pd.cut(train.buffer_months, [-99, 1, 2, 3, 6, 99],
                  labels=["<1", "1-2", "2-3", "3-6", "6+"], right=False), y,
           "2. Buffer (months of spending) vs going negative next year")

# 3. No-other-bank-transfer accounts (no transfers with any other bank)
rate_table(train.no_other_bank.map({1: "no transfers to other banks", 0: "connected"}), y,
           "3a. No-other-bank-transfer vs connected accounts")
a, b = y[train.no_other_bank == 1], y[train.no_other_bank == 0]
print("Fisher exact p =", f"{fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]:.2e}")

# 3b. Replication on a different outcome: loan default, using ONLY pre-loan data
pre_tx = tx.merge(loans[["account_id", "loan_start"]], on="account_id")
pre_tx = pre_tx[pre_tx.date < pre_tx.loan_start]
L = _lb.copy()
L["cash_only"] = (pre_tx.groupby("account_id").partner_bank.nunique().reindex(L.index).fillna(0) == 0).astype(int)
L["log_cov"] = np.log(L["cov"].clip(lower=0.01))
rate_table(L.cash_only.map({1: "no transfers to other banks", 0: "connected"}), L.d, "3b. Loan default by no-other-bank-transfer (pre-loan data)")
lm = sm.Logit(L.d, sm.add_constant(L[["cash_only", "log_cov"]])).fit(disp=0)
ci = np.exp(lm.conf_int())
print(f"Logit, controlling buffer/payment: no-other-bank-transfer OR = {np.exp(lm.params.cash_only):.1f} "
      f"(95% CI {ci.loc['cash_only', 0]:.1f}-{ci.loc['cash_only', 1]:.1f}), p = {lm.pvalues.cash_only:.1e}")

# 4. Loan buffer coverage
rate_table(pd.cut(L["cov"], [-1e9, 4, 8, 16, 1e9], labels=["<4", "4-8", "8-16", "16+"], right=False), L.d,
           "4. Loan default by buffer / monthly payment (pre-loan data)")

# 5. Past negative balance
rate_table(train.went_negative_before.map({1: "went negative before", 0: "never"}), y,
           "5. Went negative before vs again next year")

# 6. Lumpy spending
q75 = train.spend_volatility.quantile(0.75)
rate_table((train.spend_volatility > q75).map({True: "top 25% lumpy", False: "rest"}), y,
           "6. Spending volatility vs going negative next year")

# 7. Risk model used to rank plans
print("\n--- 7. Risk model (logistic regression) ---")
ci = np.exp(RISK_MODEL.conf_int())
print(pd.DataFrame({"odds_ratio": np.exp(RISK_MODEL.params), "CI_low": ci[0], "CI_high": ci[1],
                    "z": RISK_MODEL.tvalues, "p": RISK_MODEL.pvalues}).round(3))
print(f"AUC on 30% held-out accounts: {TEST_AUC:.3f}")
print("Note: spend_volatility was tested and dropped (p = 0.97 once the other three were included).")
print("Note: standing_orders and cards were NOT used: they are end-of-data snapshots / issued after loans (leakage).")
