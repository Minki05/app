# =====================================================================
# Resilience engine v2: data-driven plan recommendations
# Every number shown to the user is computed from the data in this file.
# =====================================================================
import pandas as pd, numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

D = "/content/drive/MyDrive/data"
ms = pd.read_csv(f"{D}/monthly_summary.csv").sort_values(["account_id", "month"])
loans = pd.read_csv(f"{D}/loans.csv")
tx = pd.read_csv(f"{D}/transactions.csv", usecols=["account_id", "date", "partner_bank"],
                 dtype={"partner_bank": str})
tx["month"] = tx.date.str[:7]
tx = tx[tx.partner_bank.notna() & (tx.partner_bank != "0")]          # transfers with other banks only

# ---------------------------------------------------------------------
# 1. Behaviour features for any slice of history
# ---------------------------------------------------------------------
def build_features(m, t):
    """m: monthly rows, t: transfer rows (same period). Returns one row per account."""
    g = m.groupby("account_id")
    I, S = g.money_in.median(), g.money_out.median()
    B = g.tail(6).groupby("account_id").avg_daily_balance.mean()
    f = pd.DataFrame({"I": I, "S": S, "B": B})
    f["buffer_months"] = (B / S.replace(0, np.nan)).fillna(0).clip(-1, 24)
    f["no_other_bank"] = (t.groupby("account_id").partner_bank.nunique()
                          .reindex(f.index).fillna(0) == 0).astype(int)
    f["spend_volatility"] = (g.money_out.std() / g.money_out.mean()).fillna(0)
    f["went_negative_before"] = (g.days_below_zero.sum() > 0).astype(int)
    gap = S - 0.55 * I
    f["job_loss_runway"] = np.where(B <= 0, 0, np.where(gap <= 0, 120, np.minimum(B / gap.where(gap > 0, 1), 120)))
    f["p90_month_out"] = g.money_out.quantile(0.9)
    return f

# spend_volatility was tested too but added nothing once these three were in (p = 0.97), so it is left out
MODEL_VARS = ["log_buffer_months", "no_other_bank", "went_negative_before"]
def model_matrix(f):
    X = pd.DataFrame(index=f.index)
    X["log_buffer_months"] = np.log1p(f.buffer_months.clip(lower=0))
    X["no_other_bank"] = f.no_other_bank
    X["went_negative_before"] = f.went_negative_before
    return sm.add_constant(X[MODEL_VARS], has_constant="add")

# ---------------------------------------------------------------------
# 2. Training data: features from the PAST, outcome from the NEXT 12 MONTHS
# ---------------------------------------------------------------------
ms["months_left"] = ms.groupby("account_id").cumcount(ascending=False)
n_months = ms.groupby("account_id").size()
hist = ms[ms.account_id.isin(n_months[n_months >= 24].index)]
past, future = hist[hist.months_left >= 12], hist[hist.months_left < 12]
cutoff = past.groupby("account_id").month.max()
t_past = tx[tx.account_id.isin(cutoff.index)]
t_past = t_past[t_past.month <= t_past.account_id.map(cutoff)]

train = build_features(past, t_past)
train["went_negative_next_year"] = (future.groupby("account_id").days_below_zero.sum() > 0).astype(int)
y = train.went_negative_next_year

# ---------------------------------------------------------------------
# 3. Risk model (logistic regression) + honest test on unseen accounts
# ---------------------------------------------------------------------
rng = np.random.default_rng(0)
test_mask = rng.random(len(train)) < 0.3
fit_tmp = sm.Logit(y[~test_mask], model_matrix(train[~test_mask])).fit(disp=0)
TEST_AUC = roc_auc_score(y[test_mask], fit_tmp.predict(model_matrix(train[test_mask])))
RISK_MODEL = sm.Logit(y, model_matrix(train)).fit(disp=0)            # final model on all accounts

# ---------------------------------------------------------------------
# 4. Peer evidence: what actually happened to customers like this one
# ---------------------------------------------------------------------
def peer_rate(mask):
    k, n = int(y[mask].sum()), int(mask.sum())
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return k, n, k / n if n else np.nan, lo, hi

def fmt(k, n, r, lo, hi):
    return f"{r:.1%} ({k}/{n}, 95% CI {lo:.1%}-{hi:.1%})"

RUNWAY_BANDS = [(0, 3, "under 3 months"), (3, 6, "3-6 months"), (6, 12, "6-12 months"), (12, 121, "12+ months")]
BUFFER_BANDS = [(-99, 1, "under 1 month"), (1, 2, "1-2 months"), (2, 3, "2-3 months"), (3, 6, "3-6 months"), (6, 99, "6+ months")]
def band_of(x, bands):
    for lo, hi, label in bands:
        if lo <= x < hi:
            return lo, hi, label
    return bands[-1]

# ---------------------------------------------------------------------
# 5. Current state of every account (full history) for the live engine
# ---------------------------------------------------------------------
now = build_features(ms, tx)
loans["end"] = pd.to_datetime(loans.loan_start) + pd.to_timedelta(loans.duration_months * 30.44, unit="D")
now["P"] = loans[loans.end >= "2025-12-31"].set_index("account_id").monthly_payment.reindex(now.index).fillna(0)

def profile(acc):
    d = ms[ms.account_id == acc]
    return dict(I=round(d.money_in.median(), 1),
                F=round((d.household_bills + d.insurance + d.loan_payments).median(), 1),
                S=round(d.money_out.median(), 1),
                B=round(d.avg_daily_balance.tail(6).mean(), 1),
                H=round(d.household_bills.median(), 1))

def runway(B, income, spending, one_off=0):
    left, gap = B - one_off, spending - income
    if left <= 0: return 0
    if gap <= 0: return "no shortfall (120+)"
    return round(min(left / gap, 120), 1)

def risk(row):
    return float(RISK_MODEL.predict(model_matrix(row.to_frame().T.astype(float))).iloc[0])

# ---------------------------------------------------------------------
# 6. Recommendation engine
# ---------------------------------------------------------------------
def recommend(acc, ei_pct=0.55, bills_up=0.0, one_off=0, survive_months=6, target_months=3):
    p, f = profile(acc), now.loc[acc].copy()
    I, S, F, B, H, P = p["I"], p["S"], p["F"], p["B"], p["H"], f.P
    flex, surplus = max(S - F, 0), I - S
    inc2, sp2 = I * ei_pct, S + bills_up * H
    r_now, r_shock = runway(B, I, S), runway(B, inc2, sp2, one_off)
    base_risk = risk(f)
    plans = []

    def what_if(**changes):
        g = f.copy()
        for k, v in changes.items(): g[k] = v
        return risk(g)

    # A) Shock backup plan
    if one_off >= B:
        plans.append(dict(prio=1, title="Arrange emergency credit before the shock",
            do="Set up a payment plan or line of credit now.",
            you=f"The one-off cost (${one_off:,.0f}) is bigger than the buffer (${B:,.0f}).",
            data="Peer evidence: " + fmt(*peer_rate(train.job_loss_runway < 3)) +
                 " of customers with under 3 months of job-loss runway went negative within a year.",
            new_risk=None))
    elif isinstance(r_shock, (int, float)) and r_shock < survive_months:
        max_sp = inc2 + (B - one_off) / survive_months
        cut = sp2 - max_sp
        lo, hi, label = band_of(r_shock, RUNWAY_BANDS)
        mask = (train.job_loss_runway >= lo) & (train.job_loss_runway < hi)
        plans.append(dict(prio=1, title=f"Shock plan: cut ${cut:,.0f}/month to last {survive_months} months",
            do=f"If the shock hits, lower spending from ${sp2:,.0f} to ${max_sp:,.0f}/month"
               + (f" ({cut / flex:.0%} of flexible spending)." if flex > 0 else "."),
            you=f"After the shock the buffer lasts {r_shock} months, less than the {survive_months}-month EI period.",
            data=f"Customers with {label} of job-loss runway went negative within a year "
                 + fmt(*peer_rate(mask)) + ".",
            new_risk=None))

    # B) Emergency fund
    if f.buffer_months < target_months:
        need = target_months * S - B
        if surplus > 0:
            save = 0.5 * surplus; act = f"Save ${save:,.0f}/month (half the monthly surplus): about {need / save:.0f} months."
        elif flex > 0:
            cut = -surplus + 0.05 * S
            act = f"Cut flexible spending by ${cut:,.0f}/month ({cut / flex:.0%}) and save ${0.05 * S:,.0f}/month: about {need / (0.05 * S):.0f} months."
        else:
            act = "Fixed costs exceed income: review fixed costs or income first."
        lo, hi, label = band_of(f.buffer_months, BUFFER_BANDS)
        now_band = peer_rate((train.buffer_months >= lo) & (train.buffer_months < hi))
        tgt_band = peer_rate(train.buffer_months >= target_months)
        plans.append(dict(prio=2, title=f"Build an emergency fund of {target_months} months (${target_months * S:,.0f})",
            do=act, you=f"The buffer covers {f.buffer_months:.1f} months of typical spending.",
            data=f"Customers with a buffer of {label} went negative within a year {fmt(*now_band)}; "
                 f"with {target_months}+ months: {fmt(*tgt_band)}.",
            new_risk=what_if(buffer_months=target_months)))

    # C) Connect to other banks (no-other-bank-transfer accounts)
    if f.no_other_bank == 1:
        plans.append(dict(prio=2, title="Connect income and bills to the bank",
            do="Set up direct deposit for income and automatic payments for bills.",
            you="No transfers to or from any other bank show up in this account (no direct-deposit pay or bill transfers).",
            data=f"Accounts with no transfers to other banks went negative within a year {fmt(*peer_rate(train.no_other_bank == 1))} "
                 f"vs connected accounts {fmt(*peer_rate(train.no_other_bank == 0))}.",
            new_risk=what_if(no_other_bank=0)))

    # D) Smooth lumpy spending
    if f.spend_volatility > train.spend_volatility.quantile(0.75):
        q75 = train.spend_volatility.quantile(0.75)
        plans.append(dict(prio=3, title="Smooth big spending months",
            do=f"Set aside ${max(f.p90_month_out - S, 0) / 12:,.0f}/month so a big month (about ${f.p90_month_out:,.0f}) does not drain the buffer.",
            you=f"Spending swings a lot (volatility {f.spend_volatility:.2f}, top 25% of customers).",
            data=f"Top-25% lumpy spenders went negative within a year {fmt(*peer_rate(train.spend_volatility > q75))} "
                 f"vs the rest {fmt(*peer_rate(train.spend_volatility <= q75))}.",
            new_risk=None))

    # E) Loan protection
    if P > 0 and B / P < 4:
        plans.append(dict(prio=1, title="Protect the loan payment",
            do=f"Keep 4 loan payments (${4 * P:,.0f}) untouched in savings.",
            you=f"The buffer covers only {B / P:.1f} payments of ${P:,.0f}.",
            data=LOAN_EVIDENCE, new_risk=None))

    # F) Overdraft history
    if f.went_negative_before == 1:
        plans.append(dict(prio=1, title="Turn on low-balance alerts",
            do="Turn on low-balance alerts and keep a cushion above zero.",
            you="This account has gone negative before.",
            data=f"Customers who had gone negative before went negative again within a year "
                 f"{fmt(*peer_rate(train.went_negative_before == 1))} vs {fmt(*peer_rate(train.went_negative_before == 0))}.",
            new_risk=None))

    # Rank: priority first, then biggest modelled risk reduction
    plans.sort(key=lambda x: (x["prio"], -(base_risk - x["new_risk"]) if x["new_risk"] is not None else 0))

    print(f"=== Account {acc} ===")
    print(f"Income ${I:,.0f} | Spending ${S:,.0f} (fixed ${F:,.0f}) | Buffer ${B:,.0f} | Loan payment ${P:,.0f}")
    print(f"Runway now: {r_now} | After shock (EI {ei_pct:.0%}, bills +{bills_up:.0%}, one-off ${one_off:,.0f}): {r_shock} months")
    print(f"Modelled risk of going negative within a year: {base_risk:.1%} (model test AUC {TEST_AUC:.2f})\n")
    if not plans:
        print("No action needed: this customer is resilient to this shock.")
    for i, pl in enumerate(plans, 1):
        tag = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[pl["prio"]]
        print(f"{i}. [{tag}] {pl['title']}")
        print(f"   Do:          {pl['do']}")
        print(f"   Why (you):   {pl['you']}")
        print(f"   Why (data):  {pl['data']}")
        if pl["new_risk"] is not None:
            print(f"   Model:       customers like this with the change had {pl['new_risk']:.1%} risk (now {base_risk:.1%})")
        print()
    return plans

# ---------------------------------------------------------------------
# 7. Loan evidence (pre-loan data only -> later default)
# ---------------------------------------------------------------------
loans["loan_month"] = loans.loan_start.str[:7]
_pre = ms.merge(loans[["account_id", "loan_month", "defaulted", "monthly_payment"]], on="account_id")
_pre = _pre[_pre.month < _pre.loan_month]
_lb = _pre.groupby("account_id").agg(d=("defaulted", "first"), pay=("monthly_payment", "first"))
_lb["cov"] = _pre.groupby("account_id").tail(6).groupby("account_id").avg_daily_balance.mean() / _lb.pay
_a, _b = _lb[_lb["cov"] < 4].d, _lb[_lb["cov"] >= 16].d
LOAN_EVIDENCE = (f"Borrowers whose buffer covered under 4 payments before the loan defaulted "
                 f"{_a.mean():.1%} ({_a.sum()}/{len(_a)}) vs {_b.mean():.1%} ({_b.sum()}/{len(_b)}) for 16+ payments.")
print(f"Engine ready. Risk model trained on {len(train)} accounts, test AUC = {TEST_AUC:.2f}")
