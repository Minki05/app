"""
Run once. Reads the raw data (including the 100 MB transactions.csv) and writes small files
that the Streamlit app needs, so the app itself never has to load transactions.csv.

    python precompute.py            (expects the data in ./data)
"""
import os, json
import pandas as pd, numpy as np

D = os.environ.get("DATA_DIR", "data")
OUT = os.environ.get("OUT_DIR", "app_data")
os.makedirs(OUT, exist_ok=True)

ms = pd.read_csv(f"{D}/monthly_summary.csv").sort_values(["account_id", "month"])
loans = pd.read_csv(f"{D}/loans.csv")
tx = pd.read_csv(f"{D}/transactions.csv", usecols=["account_id", "date", "partner_bank"],
                 dtype={"partner_bank": str})
tx["month"] = tx.date.str[:7]
tx = tx[tx.partner_bank.notna() & (tx.partner_bank != "0")]   # transfers with other banks only


def build_features(m, t):
    g = m.groupby("account_id")
    I, S = g.money_in.median(), g.money_out.median()
    B = g.tail(6).groupby("account_id").avg_daily_balance.mean()
    f = pd.DataFrame({"I": I, "S": S, "B": B})
    # Spending used for planning: the median (judge's S), or the mean when the median month has no spending.
    # Without this, accounts whose median spending is $0 look like they have "0 months" of buffer.
    S_plan = S.where(S > 0, g.money_out.mean())
    f["S_plan"] = S_plan
    bm = B / S_plan.replace(0, np.nan)
    f["buffer_months"] = bm.where(S_plan > 0, np.where(B > 0, 24, 0)).clip(-1, 24)
    f["no_other_bank"] = (t.groupby("account_id").partner_bank.nunique()
                          .reindex(f.index).fillna(0) == 0).astype(int)
    f["spend_volatility"] = (g.money_out.std() / g.money_out.mean()).fillna(0)
    f["went_negative_before"] = (g.days_below_zero.sum() > 0).astype(int)
    f["neg_days_total"] = g.days_below_zero.sum()
    gap = S - 0.55 * I
    f["job_loss_runway"] = np.where(B <= 0, 0, np.where(gap <= 0, 120,
                                    np.minimum(B / gap.where(gap > 0, 1), 120)))
    f["p90_month_out"] = g.money_out.quantile(0.9)
    return f


# Training set: features from the past, outcome = negative balance in the next 12 months
ms["months_left"] = ms.groupby("account_id").cumcount(ascending=False)
n = ms.groupby("account_id").size()
hist = ms[ms.account_id.isin(n[n >= 24].index)]
past, future = hist[hist.months_left >= 12], hist[hist.months_left < 12]
cutoff = past.groupby("account_id").month.max()
t_past = tx[tx.account_id.isin(cutoff.index)]
t_past = t_past[t_past.month <= t_past.account_id.map(cutoff)]
train = build_features(past, t_past)
train["went_negative_next_year"] = (future.groupby("account_id").days_below_zero.sum() > 0).astype(int)
train.to_csv(f"{OUT}/train_features.csv")

# Current state of every account (full history)
now = build_features(ms.drop(columns="months_left"), tx)
loans["end"] = pd.to_datetime(loans.loan_start) + pd.to_timedelta(loans.duration_months * 30.44, unit="D")
now["P"] = (loans[loans.end >= "2025-12-31"].set_index("account_id").monthly_payment
            .reindex(now.index).fillna(0))
now.to_csv(f"{OUT}/now_features.csv")

# Loan evidence: pre-loan buffer / payment -> later default
loans["loan_month"] = loans.loan_start.str[:7]
pre = ms.merge(loans[["account_id", "loan_month", "defaulted", "monthly_payment"]], on="account_id")
pre = pre[pre.month < pre.loan_month].sort_values(["account_id", "month"])
lb = pre.groupby("account_id").agg(d=("defaulted", "first"), pay=("monthly_payment", "first"))
lb["cov"] = pre.groupby("account_id").tail(6).groupby("account_id").avg_daily_balance.mean() / lb.pay
a, b = lb[lb["cov"] < 4].d, lb[lb["cov"] >= 16].d
json.dump({"under4_k": int(a.sum()), "under4_n": len(a), "over16_k": int(b.sum()), "over16_n": len(b)},
          open(f"{OUT}/loan_evidence.json", "w"))

# ---------- Findings shown in the app (all from pre-loan / past data) ----------
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
fd = {}
# loan coverage bands
cov_b = pd.cut(lb["cov"], [-1e9, 4, 8, 16, 1e9], labels=["<4", "4-8", "8-16", "16+"], right=False)
fd["loan_coverage"] = [[str(k), int(v.sum()), int(v.count())] for k, v in lb.d.groupby(cov_b, observed=True)]
# no transfers with other banks before the loan -> default, + odds ratio controlling coverage
ptx = tx.merge(loans[["account_id", "loan_start"]], on="account_id")
ptx = ptx[ptx.date < ptx.loan_start]
lb["no_other_bank"] = (ptx.groupby("account_id").partner_bank.nunique().reindex(lb.index).fillna(0) == 0).astype(int)
fd["loan_no_other_bank"] = [[int(k), int(v.sum()), int(v.count())] for k, v in lb.d.groupby(lb.no_other_bank)]
lm = sm.Logit(lb.d, sm.add_constant(pd.DataFrame({"nob": lb.no_other_bank,
                                                   "lc": np.log(lb["cov"].clip(lower=0.01))}))).fit(disp=0)
ci = np.exp(lm.conf_int().loc["nob"])
fd["loan_or"] = [float(np.exp(lm.params["nob"])), float(ci[0]), float(ci[1])]
# negative balance before the loan
lb["neg_before"] = pre.groupby("account_id").days_below_zero.sum() > 0
fd["loan_neg_before"] = [int(lb.d[lb.neg_before].sum()), int(lb.neg_before.sum())]
# who the customer is vs how money moves (AUC on next-year outcome, 0.5 = coin flip)
acc_ = pd.read_csv(f"{D}/accounts.csv").set_index("account_id")
cl_ = pd.read_csv(f"{D}/clients.csv"); dp_ = pd.read_csv(f"{D}/dispositions.csv")
dist_ = pd.read_csv(f"{D}/districts.csv").set_index("district_id")
own = dp_[dp_.type == "owner"].merge(cl_, on="client_id").set_index("account_id").reindex(train.index)
yy = train.went_negative_next_year
who = {"Age": 2025 - pd.to_datetime(own.birth_date).dt.year,
       "Gender": (own.gender.str.upper().str[0] == "M").astype(int),
       "District unemployment": acc_.district_id.reindex(train.index).map(dist_.unemployment_rate_2023),
       "District salary": acc_.district_id.reindex(train.index).map(dist_.average_salary)}
how = {"No transfers with other banks": train.no_other_bank, "Went negative before": train.went_negative_before,
       "Months of savings": -train.buffer_months}
auc = lambda x: float(max(roc_auc_score(yy, x.fillna(x.median())), 1 - roc_auc_score(yy, x.fillna(x.median()))))
fd["who_auc"] = {k: auc(v) for k, v in who.items()}
fd["how_auc"] = {k: auc(v) for k, v in how.items()}
json.dump(fd, open(f"{OUT}/findings.json", "w"))

# Monthly balance candles for the chart: open = last month's close, close = month-end balance,
# low = lowest end-of-day balance, high = highest balance seen in that month's transactions
tb = pd.read_csv(f"{D}/transactions.csv", usecols=["account_id", "date", "balance"])
tb["month"] = tb.date.str[:7]
c = ms[["account_id", "month", "month_end_balance", "min_balance", "avg_daily_balance"]].merge(
    tb.groupby(["account_id", "month"]).balance.max().rename("high").reset_index(), on=["account_id", "month"], how="left")
c["close"] = c.month_end_balance
c["open"] = c.groupby("account_id").close.shift(1).fillna(c.avg_daily_balance)
c["low"] = c[["min_balance", "open", "close"]].min(axis=1)
c["high"] = c[["high", "open", "close"]].max(axis=1)
c[["account_id", "month", "open", "high", "low", "close"]].round(0).to_csv(f"{OUT}/monthly_ohlc.csv", index=False)

# Slim monthly file for the app (balance chart + profile)
ms.drop(columns="months_left").to_csv(f"{OUT}/monthly_summary.csv", index=False)
print(f"Done: {len(train)} training accounts, {len(now)} accounts in {OUT}/")
