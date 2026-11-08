"""
Discovery notebook: reproduces every finding in our write-up from the raw data.

    python discovery.py            (expects the raw CSVs in ./data, including transactions.csv)

Steps
  1. Build features from each account's PAST, outcome = balance below $0 in the NEXT 12 months.
  2. Screen 40+ candidate features one by one (logistic regression controlling for the judges' runway),
     with Benjamini-Hochberg FDR correction for multiple testing.
  3. Main finding: accounts with no transfers to/from any other bank. Rates, Fisher test, odds ratio.
  4. Replication on a different outcome: loan default, using only data from before each loan.
  5. Leakage check: a model with ALL variables ranks standing orders / cards highest, but those are
     end-of-data snapshots or issued after the loan, so they are excluded.
  6. What does NOT matter: age, gender, district.
  7. Buffer months and loan-payment coverage tables.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
D = os.environ.get("DATA_DIR", "data")


def rate_table(groups, y, title):
    t = y.groupby(groups, observed=True).agg(events="sum", n="count")
    lo, hi = proportion_confint(t.events, t.n, method="wilson")
    t["rate"] = (t.events / t.n).map("{:.1%}".format)
    t["95% CI"] = [f"{a:.1%} - {b:.1%}" for a, b in zip(lo, hi)]
    print(f"\n--- {title} ---\n{t}")


def odds_ratio(model, name):
    ci = np.exp(model.conf_int().loc[name])
    return f"OR {np.exp(model.params[name]):.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), p = {model.pvalues[name]:.1e}"


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
ms = pd.read_csv(f"{D}/monthly_summary.csv").sort_values(["account_id", "month"])
loans = pd.read_csv(f"{D}/loans.csv")
acc = pd.read_csv(f"{D}/accounts.csv")
dist = pd.read_csv(f"{D}/districts.csv")
cl = pd.read_csv(f"{D}/clients.csv")
dp = pd.read_csv(f"{D}/dispositions.csv")
cards = pd.read_csv(f"{D}/cards.csv")
so = pd.read_csv(f"{D}/standing_orders.csv")
tx = pd.read_csv(f"{D}/transactions.csv", dtype={"partner_bank": str},
                 usecols=["account_id", "date", "direction", "operation", "amount", "partner_bank"])
tx["month"] = tx.date.str[:7]

# ---------------------------------------------------------------------
# 1. Past features -> next-12-month outcome
# ---------------------------------------------------------------------
ms["months_left"] = ms.groupby("account_id").cumcount(ascending=False)
n = ms.groupby("account_id").size()
keep = n[n >= 24].index
h = ms[ms.account_id.isin(keep)]
past, future = h[h.months_left >= 12], h[h.months_left < 12]
cutoff = past.groupby("account_id").month.max()
tp = tx[tx.account_id.isin(keep)]
tp = tp[tp.month <= tp.account_id.map(cutoff)]
g = past.groupby("account_id")

X = pd.DataFrame(index=cutoff.index)
X["y"] = (future.groupby("account_id").days_below_zero.sum() > 0).astype(int)
I, S = g.money_in.median(), g.money_out.median()
B = g.tail(6).groupby("account_id").avg_daily_balance.mean()
gap = S - 0.55 * I
X["log_runway"] = np.log1p(np.where(B <= 0, 0, np.where(gap <= 0, 120, np.minimum(B / np.where(gap > 0, gap, 1), 120))))
S_plan = S.where(S > 0, g.money_out.mean())
X["buffer_months"] = (B / S_plan.replace(0, np.nan)).fillna(24).clip(-1, 24)

F = {}
tot_in, tot_out = g.money_in.sum(), g.money_out.sum()
for c in ["cash_deposits", "transfers_in", "pension_in", "interest_in"]:
    F[f"share_in_{c}"] = g[c].sum() / tot_in
for c in ["household_bills", "insurance", "loan_payments", "bank_fees", "cash_withdrawals_genuine",
          "card_withdrawals", "transfers_out_other"]:
    F[f"share_out_{c}"] = g[c].sum() / tot_out
F["cv_income"] = g.money_in.std() / g.money_in.mean()
F["cv_spending"] = g.money_out.std() / g.money_out.mean()
F["dip_depth"] = g.min_balance.mean() / g.avg_daily_balance.mean()
F["log_income"] = np.log1p(I)
F["log_buffer"] = np.log1p(B.clip(lower=0))
F["balance_trend"] = (B - g.head(6).groupby("account_id").avg_daily_balance.mean()) / g.avg_daily_balance.mean()
F["months_history"] = g.size()
F["zero_income_months"] = (past.money_in == 0).groupby(past.account_id).mean()
F["went_negative_before"] = (g.days_below_zero.sum() > 0).astype(int)
tg = tp.groupby("account_id")
F["tx_per_month"] = tg.size() / g.size()
F["avg_tx_amount"] = tg.amount.mean()
transfers = tp[tp.partner_bank.notna() & (tp.partner_bank != "0")]
F["n_other_banks"] = transfers.groupby("account_id").partner_bank.nunique()
F["no_other_bank"] = (F["n_other_banks"].reindex(X.index).fillna(0) == 0).astype(int)
F["weekend_tx_share"] = (pd.to_datetime(tp.date).dt.dayofweek >= 5).groupby(tp.account_id).mean()
a = acc.set_index("account_id")
owner = dp[dp.type == "owner"].merge(cl, on="client_id").set_index("account_id")
F["age"] = 2025 - pd.to_datetime(owner.birth_date).dt.year
F["male"] = (owner.gender.str.upper().str[0] == "M").astype(int)
F["joint_account"] = (dp.groupby("account_id").size() >= 2).astype(int)
dd = dist.set_index("district_id")
for c in ["inhabitants", "urban_ratio_pct", "average_salary", "unemployment_rate_2023", "crimes_2023"]:
    F[f"district_{c}"] = a.district_id.map(dd[c])
F = pd.DataFrame(F).reindex(X.index).fillna(0)
y = X.y
print(f"Accounts: {len(X)} | went below $0 in the next 12 months: {y.mean():.1%}")

# ---------------------------------------------------------------------
# 2. Feature screen with FDR correction
# ---------------------------------------------------------------------
rows = []
for c in F.columns:
    x = F[c].astype(float)
    if x.std() == 0:
        continue
    Z = pd.DataFrame({"x": (x - x.mean()) / x.std(), "log_runway": X.log_runway})
    try:
        r = sm.Logit(y, sm.add_constant(Z)).fit(disp=0, maxiter=300)
        sep = abs(r.params["x"]) > 8
        rows.append((c, np.exp(r.params["x"]), r.tvalues["x"], r.pvalues["x"], "perfect separation" if sep else ""))
    except Exception:
        rows.append((c, np.nan, np.nan, np.nan, "did not converge"))
R = pd.DataFrame(rows, columns=["feature", "OR_per_SD", "z", "p", "note"]).dropna(subset=["p"])
R["q_FDR"] = multipletests(R.p, method="fdr_bh")[1]
print(f"\n=== 2. Screen of {len(R)} features (each controlling for judge runway), FDR-corrected ===")
print(R.sort_values("p").round(4).to_string(index=False))

# ---------------------------------------------------------------------
# 3. Main finding: no transfers with any other bank
# ---------------------------------------------------------------------
print("\n=== FINDING 1. No transfers to/from any other bank ===")
grp = F.no_other_bank.map({1: "no transfers with other banks", 0: "has transfers"})
rate_table(grp, y, "Went below $0 in the next 12 months")
k1, n1 = y[F.no_other_bank == 1].sum(), (F.no_other_bank == 1).sum()
k0, n0 = y[F.no_other_bank == 0].sum(), (F.no_other_bank == 0).sum()
print("Fisher exact p =", f"{fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1]:.1e}")
m = sm.Logit(y, sm.add_constant(pd.DataFrame({"no_other_bank": F.no_other_bank,
                                               "log_buffer_months": np.log1p(X.buffer_months.clip(lower=0)),
                                               "went_negative_before": F.went_negative_before}))).fit(disp=0)
print("Controlling for buffer and past negatives:", odds_ratio(m, "no_other_bank"))
nob = F.index[F.no_other_bank == 1]
uses = ms[ms.account_id.isin(nob)].groupby("account_id")[["card_withdrawals", "household_bills", "insurance"]].sum()
print(f"Note: {int((uses.sum(axis=1) > 0).sum())} of {len(nob)} of these accounts still use a card or pay bills/"
      f"insurance at the bank, so this means 'no transfers with other banks', NOT 'cash only'.")

# ---------------------------------------------------------------------
# 4. Replication: loan default using pre-loan data only
# ---------------------------------------------------------------------
print("\n=== FINDING 1 (replication). Loan default, pre-loan data only ===")
loans["loan_month"] = loans.loan_start.str[:7]
pre = ms.merge(loans[["account_id", "loan_month", "defaulted", "monthly_payment"]], on="account_id")
pre = pre[pre.month < pre.loan_month].sort_values(["account_id", "month"])
L = pre.groupby("account_id").agg(defaulted=("defaulted", "first"), payment=("monthly_payment", "first"),
                                  neg=("days_below_zero", "sum"))
L["B"] = pre.groupby("account_id").tail(6).groupby("account_id").avg_daily_balance.mean()
L["coverage"] = L.B / L.payment
ptx = tx.merge(loans[["account_id", "loan_start"]], on="account_id")
ptx = ptx[(ptx.date < ptx.loan_start) & ptx.partner_bank.notna() & (ptx.partner_bank != "0")]
L["no_other_bank"] = (ptx.groupby("account_id").partner_bank.nunique().reindex(L.index).fillna(0) == 0).astype(int)
rate_table(L.no_other_bank.map({1: "no transfers with other banks", 0: "has transfers"}), L.defaulted,
           "Loan default by transfers before the loan")
lm = sm.Logit(L.defaulted, sm.add_constant(pd.DataFrame({"no_other_bank": L.no_other_bank,
                                                         "log_coverage": np.log(L.coverage.clip(lower=0.01))}))).fit(disp=0)
print("Controlling for buffer / loan payment:", odds_ratio(lm, "no_other_bank"))

# ---------------------------------------------------------------------
# 5. Leakage check
# ---------------------------------------------------------------------
print("\n=== 5. Leakage check (loan default, all variables) ===")
num = [c for c in ms.columns if c not in ["account_id", "month", "months_left"]]
gp = pre.groupby("account_id")
LX = gp[[c for c in num if c in pre.columns]].median().add_suffix("_med")
LX["n_standing_orders"] = so.groupby("account_id").size().reindex(LX.index).fillna(0)
LX["has_card"] = LX.index.isin(cards.merge(dp, on="disposition_id").account_id).astype(int)
LX = LX.fillna(0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rf = RandomForestClassifier(400, min_samples_leaf=5, random_state=0, n_jobs=-1)
yd = L.defaulted.reindex(LX.index)
rf.fit(LX, yd)
imp = pd.Series(rf.feature_importances_, LX.columns).sort_values(ascending=False)
print("Top features with ALL variables:\n", imp.head(5).round(3).to_string())
c = cards.merge(dp, on="disposition_id").merge(loans[["account_id", "loan_start"]], on="account_id")
print(f"Cards issued AFTER the loan: {(c.issued >= c.loan_start).mean():.0%}. "
      "standing_orders.csv is an undated end-of-data snapshot. Both are excluded (leakage).")
clean = LX.drop(columns=["n_standing_orders", "has_card"])
print(f"Cross-validated AUC without them: {cross_val_score(rf, clean, yd, cv=cv, scoring='roc_auc').mean():.3f}")

# ---------------------------------------------------------------------
# 6. What does not matter
# ---------------------------------------------------------------------
print("\n=== FINDING 5. How money moves beats who you are (AUC, 0.5 = coin flip) ===")
for c in ["age", "male", "district_unemployment_rate_2023", "district_average_salary", "district_urban_ratio_pct"]:
    auc = roc_auc_score(y, F[c])
    print(f"  {c:35s} {max(auc, 1 - auc):.3f}")
for c in ["no_other_bank", "went_negative_before", "log_buffer"]:
    auc = roc_auc_score(y, F[c])
    print(f"  {c:35s} {max(auc, 1 - auc):.3f}   (for comparison)")

# ---------------------------------------------------------------------
# 7. Buffer and loan coverage tables
# ---------------------------------------------------------------------
rate_table(pd.cut(X.buffer_months, [-99, 1, 2, 3, 6, 99], labels=["<1", "1-2", "2-3", "3-6", "6+"], right=False),
           y, "FINDING 2. Under one month of savings is a cliff")
rate_table(pd.cut(L.coverage, [-1e9, 4, 8, 16, 1e9], labels=["<4", "4-8", "8-16", "16+"], right=False),
           L.defaulted, "FINDING 3. Savings vs loan payment predicts default (pre-loan data)")

# ---------------------------------------------------------------------
# FINDING 4. Overdrafts repeat
# ---------------------------------------------------------------------
rate_table(F.went_negative_before.map({1: "went negative before", 0: "never"}), y,
           "FINDING 4. Went negative before -> again in the next 12 months")
nb = L[L.neg > 0]
print(f"Borrowers with a negative balance before the loan: {int(nb.defaulted.sum())} of {len(nb)} later defaulted "
      "(small group: a warning sign, not a rule).")

# ---------------------------------------------------------------------
# FINDING 6. The judges' runway works; our model adds to it
# ---------------------------------------------------------------------
rw = np.expm1(X.log_runway)
rate_table(pd.cut(rw, [-1, 3, 6, 12, 121], labels=["<3 mo", "3-6 mo", "6-12 mo", "12+ mo"], right=False), y,
           "FINDING 6. Judge job-loss runway vs going below $0 next year")
M = pd.DataFrame({"log_buffer_months": np.log1p(X.buffer_months.clip(lower=0)), "no_other_bank": F.no_other_bank,
                  "went_negative_before": F.went_negative_before})
test = np.random.default_rng(0).random(len(M)) < 0.3
mm = sm.Logit(y[~test], sm.add_constant(M[~test])).fit(disp=0)
pt = mm.predict(sm.add_constant(M[test]))
mm2 = sm.Logit(y[~test], sm.add_constant(M.drop(columns="went_negative_before")[~test])).fit(disp=0)
pt2 = mm2.predict(sm.add_constant(M.drop(columns="went_negative_before")[test]))
print(f"Risk model AUC on 30% held-out accounts: {roc_auc_score(y[test], pt):.3f} "
      f"(without past negatives: {roc_auc_score(y[test], pt2):.3f})")
q = pd.qcut(pt.rank(method="first"), 5, labels=False)
cal = pd.DataFrame({"predicted": pt, "actual": y[test]}).groupby(q).mean()
print("Calibration by predicted-risk quintile:\n", cal.round(3))
