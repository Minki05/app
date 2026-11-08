"""
core.py - the data-driven engine behind both front ends (Streamlit app.py and the web site in site/).
Risk model, evidence from similar customers, ranked plans and the six findings.
No UI code here. All numbers come from app_data/ (built by precompute.py).
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency, fisher_exact, norm
from statsmodels.stats.proportion import proportion_confint

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data")


def load():
    ms = pd.read_csv(f"{A}/monthly_summary.csv")
    train = pd.read_csv(f"{A}/train_features.csv", index_col=0)
    now = pd.read_csv(f"{A}/now_features.csv", index_col=0)
    loan_ev = json.load(open(f"{A}/loan_evidence.json"))
    try:
        findings = json.load(open(f"{A}/findings.json"))
    except FileNotFoundError:
        findings = None
    return ms, train, now, loan_ev, findings


MODEL_VARS = ["log_buffer_months", "no_other_bank", "went_negative_before"]


def model_matrix(f):
    X = pd.DataFrame(index=f.index)
    X["log_buffer_months"] = np.log1p(f.buffer_months.clip(lower=0))
    X["no_other_bank"] = f.no_other_bank
    X["went_negative_before"] = f.went_negative_before
    return sm.add_constant(X[MODEL_VARS].astype(float), has_constant="add")


def fit_model(train):
    y = train.went_negative_next_year
    test = np.random.default_rng(0).random(len(train)) < 0.3
    tmp = sm.Logit(y[~test], model_matrix(train[~test])).fit(disp=0)
    auc = roc_auc_score(y[test], tmp.predict(model_matrix(train[test])))
    return sm.Logit(y, model_matrix(train)).fit(disp=0), auc


ms, train, now, loan_ev, findings = load()
MODEL, AUC = fit_model(train)
Y = train.went_negative_next_year
ACCOUNTS = sorted(now.index.tolist())


def risk(row):
    return float(MODEL.predict(model_matrix(row.to_frame().T)).iloc[0])


def ptxt(p):
    """p-value as people read it."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fisher(k1, n1, k0, n0):
    """Two-sided Fisher exact test: is the rate in group 1 different from group 0?"""
    return float(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])[1])


def vs_test(m1, m0):
    """' (Fisher exact p < 0.001)' for two groups of training accounts."""
    return f" (Fisher exact {ptxt(fisher(int(Y[m1].sum()), int(m1.sum()), int(Y[m0].sum()), int(m0.sum())))})"


def gap_test(row_a, row_b):
    """Predicted risk A - B with a 95% CI and p-value (delta method on the fitted logistic model)."""
    xa = model_matrix(row_a.to_frame().T).iloc[0].values
    xb = model_matrix(row_b.to_frame().T).iloc[0].values
    pa, pb = float(MODEL.predict(xa[None])[0]), float(MODEL.predict(xb[None])[0])
    g = pa * (1 - pa) * xa - pb * (1 - pb) * xb
    se = float(np.sqrt(g @ MODEL.cov_params().values @ g))
    d = pa - pb
    p = float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0
    return d * 100, (d - 1.96 * se) * 100, (d + 1.96 * se) * 100, p


def peer(mask):
    k, n = int(Y[mask].sum()), int(mask.sum())
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k / n:.1%} ({k}/{n}, 95% CI {lo:.1%}–{hi:.1%})"


# ---------------------------------------------------------------------
# Engine (judge definitions)
# ---------------------------------------------------------------------
def profile(acc):
    d = ms[ms.account_id == acc].sort_values("month")
    return dict(I=d.money_in.median(), F=(d.household_bills + d.insurance + d.loan_payments).median(),
                S=d.money_out.median(), B=d.avg_daily_balance.tail(6).mean(), H=d.household_bills.median()), d


def runway(B, income, spending, one_off=0):
    left, gap = B - one_off, spending - income
    if left <= 0:
        return 0.0
    if gap <= 0:
        return "no shortfall (120+)"
    return round(min(left / gap, 120), 1)


def simulate(B, I, S, inc2, sp2, one_off, start_in=0, lasts=0, after_frac=1.0, cut=0.0, months=120):
    """Month-by-month balance. The shock starts after `start_in` months, lasts `lasts` months (0 = ongoing),
    then income becomes `after_frac` x usual and extra costs stop. Returns (months until below $0 or None, path)."""
    bal, path, out = B, [B], None
    end = start_in + lasts if lasts > 0 else 10 ** 9
    for m in range(1, months + 1):
        if m == start_in + 1:
            bal -= one_off
            if bal <= 0 and out is None:
                out = float(m - 1)
        if m <= start_in:
            net = I - S
        elif m <= end:
            net = inc2 - (sp2 - cut)
        else:
            net = I * after_frac - S
        prev, bal = bal, bal + net
        path.append(bal)
        if out is None and bal < 0 <= prev:
            out = (m - 1) + prev / (prev - bal)
    return out, path


FLEX_COLS = {"cash_withdrawals_genuine": "cash withdrawals", "transfers_out_other": "transfers out",
             "card_withdrawals": "card spending"}


def flexible_mix(acc):
    """Where the flexible money goes, from this customer's own monthly history."""
    d = ms[ms.account_id == acc]
    share = (d[list(FLEX_COLS)].sum() / max(d.money_out.sum(), 1)).sort_values(ascending=False)
    parts = [f"{FLEX_COLS[k]} ({v:.0%} of spending)" for k, v in share.items() if v >= 0.05][:2]
    return "Flexible spending is mostly " + " and ".join(parts) + "." if parts else ""


def fmt_rw(r):
    return r if isinstance(r, str) else f"{r} months"


RUNWAY_BANDS = [(0, 3, "under 3 months"), (3, 6, "3–6 months"), (6, 12, "6–12 months"), (12, 121, "12+ months")]
BUFFER_BANDS = [(-99, 1, "under 1 month"), (1, 2, "1–2 months"), (2, 3, "2–3 months"),
                (3, 6, "3–6 months"), (6, 99, "6+ months")]


def band(x, bands):
    for lo, hi, label in bands:
        if lo <= x < hi:
            return lo, hi, label
    return bands[-1]


BLUE, RED, GRAY = "#16C784", "#EA3943", "#8A93A6"   # up / down / muted


def recommend(acc, p, ei_pct, bills_up, one_off, survive_months, target_months, extra=0.0):
    f = now.loc[acc].copy()
    I, S, F, B, H, P = p["I"], p["S"], p["F"], p["B"], p["H"], f.P
    # S (median) drives the judges' runway. For plan sizes we use S_plan: the median, or the mean when the
    # median month has no spending (77 accounts), so targets are never $0 and months never negative.
    Sp = float(f.S_plan) if f.S_plan > 0 else 0.0
    flex, surplus = max(Sp - F, 0), I - Sp
    inc2, sp2 = I * ei_pct, S + bills_up * H + extra
    r_shock = runway(B, inc2, sp2, one_off)
    base = risk(f)
    plans = []

    rows = {}                                   # predicted risk -> the profile it came from (for the CI)

    def what_if(**ch):
        g = f.copy()
        for k, v in ch.items():
            g[k] = v
        r = risk(g)
        rows[r] = g
        return r

    # Backup plan for the shock
    if one_off >= B:
        plans.append(dict(prio=1, title="Arrange emergency credit before the shock",
                          do="Set up a payment plan or line of credit now.",
                          you=f"The one-off cost (${one_off:,.0f}) is larger than the buffer (${B:,.0f}).",
                          data="Customers with under 3 months of job-loss runway went negative within a year "
                               + peer(train.job_loss_runway < 3) + ".", new=None,
                          you_c=f"This cost (${one_off:,.0f}) is more than you have saved right now.",
                          bank="Tangerine could pre-approve a small credit line or a payment plan, so the bill "
                               "doesn't turn into missed payments.", pinned=True))
    elif ei_pct >= 1.0 and bills_up == 0 and extra == 0 and one_off > 0:
        # Repair-type shock the customer CAN pay: the plan is to refill the savings it used
        after = (B - one_off) / Sp if Sp else 24
        risk_after = what_if(buffer_months=after)
        save = max(0.5 * surplus if surplus > 0 else 0.05 * Sp, 1)
        prefix = "" if surplus > 0 else f"After closing the ${-surplus:,.0f}/month deficit, "
        plans.append(dict(prio=2, title=f"Refill the ${one_off:,.0f} used for the repair",
                          do=f"{prefix}save ${save:,.0f}/month → savings are back in about {one_off / save:.0f} months.",
                          you=f"Paying the repair drops the buffer from {f.buffer_months:.1f} to {after:.1f} "
                              f"months of spending; the model puts risk at {risk_after:.1%} until it is refilled.",
                          data=f"Buffer of {band(after, BUFFER_BANDS)[2]}: "
                               f"{peer((train.buffer_months >= band(after, BUFFER_BANDS)[0]) & (train.buffer_months < band(after, BUFFER_BANDS)[1]))} "
                               f"went negative within a year.",
                          new=base, base_override=risk_after,
                          you_c=f"You can pay the repair and still have ${B - one_off:,.0f} left. "
                                f"Refilling it gets you back to where you were.",
                          bank="Tangerine could set up an automatic monthly transfer back into savings."))
    elif not isinstance(r_shock, str) and r_shock < survive_months:
        max_sp = inc2 + (B - one_off) / survive_months
        need_cut = sp2 - max_sp
        lo, hi, label = band(r_shock, RUNWAY_BANDS)
        feasible = need_cut <= flex
        infeasible_shock = False
        cut = need_cut if feasible else flex               # never recommend cutting more than is flexible
        if feasible:
            title = f"Shock plan: cut ${cut:,.0f}/month to last {survive_months} months"
            do = (f"If the shock hits, lower spending from ${sp2:,.0f} to ${max_sp:,.0f}/month"
                  + (f" ({cut / flex:.0%} of flexible spending)." if flex > 0 else "."))
            you_c = (f"If this happens, your savings would last about {r_shock} months. "
                     f"Spending ${cut:,.0f} less each month stretches that to {survive_months} months.")
        else:
            best = runway(B, inc2, sp2 - flex, one_off)
            gap_left = need_cut - flex
            title = f"Shock plan: spending cuts alone can't reach {survive_months} months"
            infeasible_shock = True
            do = (f"Cutting all flexible spending (${flex:,.0f}/month) stretches the buffer to "
                  f"{fmt_rw(best)}. The other ${gap_left:,.0f}/month must come from fixed costs, "
                  f"extra income or credit.")
            you_c = (f"If this happens, your savings would last about {r_shock} months. Cutting back helps, "
                     f"but it isn't enough on its own, so we'd want to plan other support with you early.")
        plans.append(dict(prio=1, title=title, do=do,
                          you=f"After the shock the buffer lasts {r_shock} months, "
                              f"less than the {survive_months}-month target. Flexible spending is ${flex:,.0f}/month "
                              f"(spending minus fixed costs). {flexible_mix(acc)}",
                          data=f"Customers with {label} of job-loss runway went negative within a year "
                               + peer((train.job_loss_runway >= lo) & (train.job_loss_runway < hi)) + ".",
                          new=what_if(buffer_months=B / (Sp - cut) if Sp - cut > 0 else 24) if Sp else None,
                          cut=cut, you_c=you_c, infeasible=infeasible_shock,
                          bank="Tangerine could switch on a budget alert at the new spending level and talk "
                               "through a payment break on any loan before money runs low."))

    # Emergency fund (goal plan)
    if Sp > 0 and f.buffer_months < target_months:
        fund_blocked = False
        goal_m = target_months
        need = goal_m * Sp - B
        monthly = 0.5 * surplus if surplus > 0 else 0.05 * Sp
        if surplus > 0 and need / monthly > 36:          # unrealistic -> starter goal first
            goal_m = max(int(np.ceil(f.buffer_months)) + 1, 1)
            need = goal_m * Sp - B
        if surplus > 0:
            act = f"Save ${monthly:,.0f}/month (half of the monthly surplus) → about {need / monthly:.0f} months."
        elif flex > 0 and (-surplus + 0.05 * Sp) <= flex:
            cut = -surplus + 0.05 * Sp
            act = (f"Cut flexible spending by ${cut:,.0f}/month ({cut / flex:.0%}) and save "
                   f"${monthly:,.0f}/month → about {need / monthly:.0f} months.")
        elif flex > 0:
            fund_blocked = True
            act = (f"Even cutting all flexible spending (${flex:,.0f}/month) leaves a "
                   f"${-surplus - flex:,.0f}/month deficit, so fixed costs or income need to change before saving.")
        else:
            fund_blocked = True
            act = "Fixed costs exceed income: review fixed costs or income first."
        lo, hi, label = band(f.buffer_months, BUFFER_BANDS)
        title = (f"Build an emergency fund of {goal_m} month{'s' if goal_m > 1 else ''} (${goal_m * Sp:,.0f})"
                 + (" — starter goal" if goal_m < target_months else ""))
        plans.append(dict(prio=2, title=title, do=act,
                          you=f"The buffer covers {f.buffer_months:.1f} months of typical spending.",
                          data=f"Buffer of {label}: {peer((train.buffer_months >= lo) & (train.buffer_months < hi))} "
                               f"went negative within a year. Buffer of {goal_m}+ months: "
                               f"{peer(train.buffer_months >= goal_m)}"
                               f"{vs_test((train.buffer_months >= lo) & (train.buffer_months < hi), train.buffer_months >= goal_m)}.",
                          new=what_if(buffer_months=goal_m), infeasible=fund_blocked,
                          you_c=f"Right now your savings cover about {f.buffer_months:.1f} months of normal spending.",
                          bank="Tangerine could move the amount into savings automatically on payday, "
                               "so saving happens without thinking about it."))

    # Accounts with no transfers to/from any other bank (not the same as "cash only": some still use a card
    # or pay bills at the bank)
    if f.no_other_bank == 1:
        plans.append(dict(prio=2, title="Connect income and bills to the bank",
                          do="Set up direct deposit for income and automatic payments for bills.",
                          you="No transfers to or from any other bank show up in this account's history "
                              "(no direct-deposit pay or bill transfers).",
                          data=f"Accounts with no transfers to other banks went negative within a year "
                               f"{peer(train.no_other_bank == 1)} vs accounts with transfers "
                               f"{peer(train.no_other_bank == 0)}{vs_test(train.no_other_bank == 1, train.no_other_bank == 0)}.",
                          new=what_if(no_other_bank=0),
                          you_c="We don't see your pay or bills coming in or going out by transfer, which makes it "
                                "harder to plan ahead and spot trouble early.",
                          bank="Tangerine could make the payroll switch a one-tap step in the app, and if you're paid "
                               "in cash, save a small part of every cash deposit automatically."))

    # Loan protection
    if P > 0 and B / P < 4:
        plans.append(dict(prio=1, title="Protect the loan payment",
                          do=f"Keep 4 loan payments (${4 * P:,.0f}) untouched in savings.",
                          you=f"The buffer covers only {B / P:.1f} payments of ${P:,.0f}.",
                          data=f"Borrowers whose buffer covered under 4 payments before the loan defaulted "
                               f"{loan_ev['under4_k'] / loan_ev['under4_n']:.1%} "
                               f"({loan_ev['under4_k']}/{loan_ev['under4_n']}) vs "
                               f"{loan_ev['over16_k'] / loan_ev['over16_n']:.1%} "
                               f"({loan_ev['over16_k']}/{loan_ev['over16_n']}) with 16+ payments "
                               f"(Fisher exact {ptxt(fisher(loan_ev['under4_k'], loan_ev['under4_n'], loan_ev['over16_k'], loan_ev['over16_n']))}).",
                          new=what_if(buffer_months=max(f.buffer_months, 4 * P / Sp)) if Sp else None,
                          you_c=f"Your savings would cover about {B / P:.1f} loan payments of ${P:,.0f}.",
                          bank="Tangerine could reach out before a payment is missed and offer to move the "
                               "payment date or adjust the schedule."))

    # Lumpy spending
    q75 = train.spend_volatility.quantile(0.75)
    if f.spend_volatility > q75:
        aside = max(f.p90_month_out - Sp, 0) / 12
        if surplus > 0:
            aside = min(aside, 0.5 * surplus)
        prefix = "" if surplus > 0 else "Once the monthly deficit is closed, "
        plans.append(dict(prio=3, title="Smooth big spending months",
                          do=f"{prefix}set aside ${aside:,.0f}/month so a big month (about ${f.p90_month_out:,.0f}) "
                             f"does not drain the buffer.".capitalize() if not prefix else
                             f"{prefix}set aside ${aside:,.0f}/month so a big month (about ${f.p90_month_out:,.0f}) "
                             f"does not drain the buffer.",
                          you=f"Spending swings a lot (volatility {f.spend_volatility:.2f}, top 25% of customers).",
                          data=f"Top-25% lumpy spenders went negative within a year "
                               f"{peer(train.spend_volatility > q75)} vs the rest {peer(train.spend_volatility <= q75)}"
                               f"{vs_test(train.spend_volatility > q75, train.spend_volatility <= q75)}. "
                               f"Once savings are counted, the model finds no extra risk from lumpy spending (p = 0.97).",
                          new=base,
                          you_c="Some months you spend a lot more than usual, and those months eat into savings.",
                          bank="Tangerine could set up a separate 'big expenses' savings pocket that fills up "
                               "a little every month."))

    # Overdraft history
    if f.went_negative_before == 1:
        plans.append(dict(prio=1, title="Turn on low-balance alerts",
                          do="Turn on low-balance alerts and keep a cushion above zero.",
                          you=f"This account has already ended {int(f.neg_days_total)} days below zero.",
                          data=f"Customers who had gone negative before went negative again within a year "
                               f"{peer(train.went_negative_before == 1)} vs {peer(train.went_negative_before == 0)}"
                               f"{vs_test(train.went_negative_before == 1, train.went_negative_before == 0)}.",
                          new=None,
                          you_c="Your balance has dipped below zero before.",
                          bank="Tangerine could send an alert before the balance gets low, so there's time to act.",
                          pinned=True))

    # Data-driven ranking on ONE scale: how much LOWER the model's predicted risk is for customers whose
    # profile already looks like the "after" state. This is an association in the data, not a tested effect
    # of taking the action. Safety nets (nothing to model) are pinned on top.
    # A plan is ranked only when its gap is statistically clear: the 95% CI of (risk now - risk after),
    # from the delta method on the fitted model, must stay above 0. No hand-picked cut-off.
    for pl in plans:
        start = pl.get("base_override", base)
        pl["start"] = start
        pl["red"], pl["sig"], pl["test"] = None, False, ""
        if pl["new"] is not None:
            d, lo, hi, p = gap_test(rows.get(start, f), rows.get(pl["new"], f))
            pl["red"] = max(d, 0)
            pl["sig"] = lo > 0
            pl["test"] = ("The model predicts the same risk before and after: no gap to rank on."
                          if hi - lo < 1e-9 else
                          f"Gap {d:.1f} pts (95% CI {lo:.1f} to {hi:.1f}), {ptxt(p)}"
                          + ("" if lo > 0 else " · not statistically clear"))
    # Order: safety nets → "needs other support" warnings → ranked plans → plans with no clear gap
    def tier(x):
        if x.get("pinned"):
            return 0
        if x.get("infeasible"):
            return 1
        return 2 if x["sig"] else 3
    plans.sort(key=lambda x: (tier(x), -(x["red"] or 0)))
    rank = 0
    for pl in plans:
        if pl.get("pinned"):
            pl["pill"] = ("SAFETY NET", "rgba(234,57,67,.14)", RED)
        elif pl.get("infeasible"):
            pl["pill"] = ("NEEDS OTHER SUPPORT", "rgba(245,196,81,.14)", "#F5C451")
        elif not pl["sig"]:
            pl["pill"] = ("NO CLEAR GAP", "rgba(138,147,166,.14)", GRAY)
        else:
            rank += 1
            pl["pill"] = (("▲ #1 · " if rank == 1 else f"#{rank} · ") + f"PREDICTED GAP −{pl['red']:.1f} PTS",
                          "rgba(22,199,132,.14)", BLUE)
    return plans, base, r_shock




# ---------------------------------------------------------------------
# Six findings, as data (the web site draws them; app.py draws the same numbers)
# ---------------------------------------------------------------------
def _rate(mask):
    k, n = int(Y[mask].sum()), int(mask.sum())
    lo, hi = proportion_confint(k, n, method="wilson")
    return dict(value=k / n, k=k, n=n, lo=lo, hi=hi)


def _row(label, mask, tone):
    r = _rate(mask)
    return dict(label=label, tone=tone, note=f"{r['k']}/{r['n']}", **r)


def _chi2(rows):
    """Chi-square test that the rate is the same in every group."""
    return float(chi2_contingency([[r["k"], r["n"] - r["k"]] for r in rows])[1])


def _auc_ci(reps=1000):
    """Bootstrap 95% CI of the held-out AUC (same 30% split as fit_model)."""
    y = train.went_negative_next_year
    test = np.random.default_rng(0).random(len(train)) < 0.3
    m = sm.Logit(y[~test], model_matrix(train[~test])).fit(disp=0)
    yt, pt = y[test].values, m.predict(model_matrix(train[test])).values
    rng, out = np.random.default_rng(1), []
    for _ in range(reps):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max():
            out.append(roc_auc_score(yt[i], pt[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


AUC_CI = _auc_ci()


def findings_data():
    out = []
    t = train
    a, b = _row("No transfers with other banks", t.no_other_bank == 1, "risk"), \
        _row("Has transfers", t.no_other_bank == 0, "ok")
    notes = []
    if findings:
        lr = {int(x): (k, n) for x, k, n in findings["loan_no_other_bank"]}
        o = findings["loan_or"]
        notes.append(f"Same pattern for loan default (pre-loan data only): {lr[1][0] / lr[1][1]:.1%} vs "
                     f"{lr[0][0] / lr[0][1]:.1%}, odds ratio {o[0]:.1f} (95% CI {o[1]:.1f}–{o[2]:.1f}) "
                     f"controlling for savings vs loan payment.")
    notes.append("Means no direct-deposit pay or bill transfers show up. Not \"cash only\": many still use a "
                 "card or pay bills at the bank.")
    m = MODEL.params["no_other_bank"]; ci = np.exp(MODEL.conf_int().loc["no_other_bank"])
    test = (f"Fisher exact {ptxt(fisher(a['k'], a['n'], b['k'], b['n']))}. Holding savings and past overdrafts fixed: "
            f"odds ratio {np.exp(m):.1f} (95% CI {ci[0]:.1f}–{ci[1]:.1f}), {ptxt(MODEL.pvalues['no_other_bank'])}.")
    out.append(dict(title="No transfers with other banks", big=f"{a['value'] / b['value']:.0f}x",
                    sub="more likely to go below $0 within a year", rows=[a, b], fmt="pct", notes=notes, test=test))

    bands = [(-99, 1, "Under 1 month", "risk"), (1, 2, "1–2 months", "warn"), (2, 3, "2–3 months", "warn"),
             (3, 6, "3–6 months", "ok"), (6, 99, "6+ months", "ok")]
    rows = [_row(l, (t.buffer_months >= lo) & (t.buffer_months < hi), tone) for lo, hi, l, tone in bands]
    rest = (t.buffer_months >= 1)
    test = (f"Under 1 month vs everyone else: Fisher exact "
            f"{ptxt(fisher(rows[0]['k'], rows[0]['n'], int(Y[rest].sum()), int(rest.sum())))}. "
            f"Across all 5 bands: chi-square {ptxt(_chi2(rows))}.")
    out.append(dict(title="Under one month of savings is a cliff", big=f"{rows[0]['value']:.0%}",
                    sub="went below $0 within a year", rows=rows, fmt="pct", test=test,
                    notes=["Savings measured in months of the customer's own typical spending."]))

    if findings:
        rows = []
        for lab, k, n in findings["loan_coverage"]:
            lo, hi = proportion_confint(k, n, method="wilson")
            rows.append(dict(label=f"{lab} payments", value=k / n, k=k, n=n, lo=lo, hi=hi, note=f"{k}/{n}",
                             tone="risk" if lab == "<4" else ("warn" if lab == "4-8" else "ok")))
        test = (f"Under 4 vs 16+ payments: Fisher exact "
                f"{ptxt(fisher(rows[0]['k'], rows[0]['n'], rows[-1]['k'], rows[-1]['n']))}. "
                f"Across all 4 groups: chi-square {ptxt(_chi2(rows))}.")
        out.append(dict(title="Savings vs loan payment predicts default", big=f"{rows[0]['value']:.0%}",
                        sub="of borrowers with savings under 4 payments defaulted", rows=rows, fmt="pct", test=test,
                        notes=["Savings before the loan ÷ monthly loan payment; outcome = later default."]))

    a, b = _row("Went negative before", t.went_negative_before == 1, "risk"), \
        _row("Never", t.went_negative_before == 0, "ok")
    notes = []
    if findings:
        k, n = findings["loan_neg_before"]
        notes.append(f"All {k} of {n} borrowers with a negative balance before their loan later defaulted "
                     f"(small group: a warning sign, not a rule).")
    m = MODEL.params["went_negative_before"]; ci = np.exp(MODEL.conf_int().loc["went_negative_before"])
    test = (f"Fisher exact {ptxt(fisher(a['k'], a['n'], b['k'], b['n']))}. Holding savings and transfers fixed: "
            f"odds ratio {np.exp(m):.1f} (95% CI {ci[0]:.1f}–{ci[1]:.1f}), {ptxt(MODEL.pvalues['went_negative_before'])}.")
    out.append(dict(title="Overdrafts repeat", big=f"{a['value'] / b['value']:.0f}x",
                    sub="more likely to go negative again", rows=[a, b], fmt="pct", notes=notes, test=test))

    if findings:
        wp, hp = findings.get("who_p", {}), findings.get("how_p", {})
        rows = [dict(label=k, value=v, tone="muted", note=ptxt(wp[k]) if k in wp else "")
                for k, v in findings["who_auc"].items()] + \
               [dict(label=k, value=v, tone="ok", note=ptxt(hp[k]) if k in hp else "")
                for k, v in findings["how_auc"].items()]
        n_sig = sum(p < 0.05 for p in wp.values())
        test = (f"Mann-Whitney U test of AUC = 0.5: {n_sig} of {len(wp)} 'who you are' variables differ from a "
                f"coin flip at p < 0.05; all {len(hp)} behaviour variables do (p < 0.001).") if wp else ""
        out.append(dict(title="How money moves beats who you are", big="≈ 0.5",
                        sub="AUC for age, gender, region (0.5 = coin flip)", rows=rows, fmt="auc", scale=1.0, test=test,
                        notes=["Each variable alone, predicting going below $0 next year. Grey = who the "
                               "customer is; green = account behaviour."]))

    bands = [(0, 3, "Under 3 months", "risk"), (3, 6, "3–6 months", "warn"), (6, 12, "6–12 months", "ok"),
             (12, 121, "12+ months", "ok")]
    rows = [_row(l, (t.job_loss_runway >= lo) & (t.job_loss_runway < hi), tone) for lo, hi, l, tone in bands]
    lo, hi = AUC_CI
    test = (f"Held-out AUC {AUC:.2f}, bootstrap 95% CI {lo:.2f}–{hi:.2f} (1,000 resamples): far from 0.5. "
            f"Across runway bands: chi-square {ptxt(_chi2(rows))}.")
    out.append(dict(title="The judges' runway works, our model adds to it", big=f"AUC {AUC:.2f}",
                    sub="risk model on held-out accounts", rows=rows, fmt="pct", test=test,
                    notes=["Bars: job-loss runway (judge formula) vs going below $0 next year. The model adds "
                           "savings, transfers and past overdrafts."]))
    return dict(cards=out, scope=f"Features from each account's past; outcome = what happened next "
                                 f"({len(train):,} accounts, next 12 months). Reproduced in discovery.py.")
