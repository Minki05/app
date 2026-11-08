"""Resilience engine: account profiles, shock runway, plans, peer comparison.

All judging numbers follow the README definitions exactly. Anything that goes
beyond them (plan maths, peer comparison, sensitivity) is named separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

def _data_dir() -> Path:
    """Raw CSVs: $RR_DATA, else site/data, else ../data (the repo's data folder)."""
    import os
    here = Path(__file__).parent
    for c in (os.environ.get("RR_DATA"), here / "data", here.parent / "data"):
        if c and (Path(c) / "loans.csv").exists():
            return Path(c)
    return here.parent / "app_data"          # no raw CSVs: run from the repo's app_data (see web_app.Store)


DATA_DIR = _data_dir()
CACHE_DIR = Path(__file__).parent / "cache"
RUNWAY_CAP = 120.0

OUT_COMPONENTS = {
    "household_bills": "Household Bills",
    "insurance": "Insurance",
    "loan_payments": "Loan Payments",
    "bank_fees": "Bank Fees",
    "cash_withdrawals_genuine": "Cash Withdrawals",
    "card_withdrawals": "Card Withdrawals",
    "transfers_out_other": "Other Transfers",
}
IN_COMPONENTS = {
    "cash_deposits": "Cash Deposits",
    "transfers_in": "Transfers In",
    "pension_in": "Pension",
    "interest_in": "Interest",
}
FIXED_COMPONENTS = ["household_bills", "insurance", "loan_payments"]


# ---------------------------------------------------------------- loading
def load_monthly() -> pd.DataFrame:
    ms = pd.read_csv(DATA_DIR / "monthly_summary.csv")
    return ms.sort_values(["account_id", "month"], kind="mergesort").reset_index(drop=True)


def load_dispositions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "dispositions.csv")


def load_clients() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "clients.csv", dtype={"birth_date": "string"})


def load_loans() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "loans.csv", parse_dates=["loan_start"])


def file_signature(path: Path) -> str:
    """Size + modification time; changes whenever the file is replaced or edited."""
    st_ = path.stat()
    return f"{st_.st_size}-{st_.st_mtime_ns}"


def app_data_signature() -> str:
    """Files the app reads (the app no longer loads transactions or client ages)."""
    return "|".join(file_signature(DATA_DIR / f) for f in ("monthly_summary.csv", "loans.csv"))


def data_signature() -> str:
    return "|".join(file_signature(DATA_DIR / f) for f in
                    ("monthly_summary.csv", "loans.csv", "transactions.csv", "dispositions.csv", "clients.csv"))


def load_transactions() -> pd.DataFrame:
    """Transactions with a parquet cache (the CSV is ~100 MB).

    The cache is rebuilt whenever transactions.csv changes, so a new monthly
    summary is never paired with stale transactions.
    """
    src = DATA_DIR / "transactions.csv"
    pq = CACHE_DIR / "transactions.parquet"
    sig_file = CACHE_DIR / "transactions.signature"
    sig = file_signature(src)
    if pq.exists() and sig_file.exists() and sig_file.read_text() == sig:
        return pd.read_parquet(pq)
    tx = pd.read_csv(
        src,
        dtype={"partner_bank": "string", "partner_account": "string", "purpose": "string"},
        parse_dates=["date"],
    )
    CACHE_DIR.mkdir(exist_ok=True)
    tx.to_parquet(pq, index=False)
    sig_file.write_text(sig)
    return tx


# ---------------------------------------------------------------- profiles
def running_loan_payment(loans: pd.DataFrame) -> pd.Series:
    """P: monthly_payment if the loan is still running at the end of 2025 (status C/D)."""
    running = loans[loans["status"].isin(["C", "D"])]
    return running.set_index("account_id")["monthly_payment"]


def compute_profiles(ms: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    ms = ms.sort_values(["account_id", "month"], kind="mergesort")
    fixed = ms[FIXED_COMPONENTS].sum(axis=1)  # sum per month first, then median
    g = ms.groupby("account_id", sort=True)
    prof = pd.DataFrame(
        {
            "I": g["money_in"].median(),
            "S": g["money_out"].median(),
            "F": fixed.groupby(ms["account_id"]).median(),
            "H": g["household_bills"].median(),
            "B": g["avg_daily_balance"].apply(lambda s: s.iloc[-6:].mean()),
            "n_months": g.size(),
            "first_month": g["month"].first(),
            "last_month": g["month"].last(),
            "latest_balance": g["month_end_balance"].last(),
            # alternative definitions, for sensitivity only
            "I_12": g["money_in"].apply(lambda s: s.iloc[-12:].median()),
            "S_12": g["money_out"].apply(lambda s: s.iloc[-12:].median()),
        }
    )
    prof["P"] = running_loan_payment(loans).reindex(prof.index).fillna(0.0)
    loan_status = loans.set_index("account_id")["status"]
    prof["loan_status"] = loan_status.reindex(prof.index)
    return prof


# ---------------------------------------------------------------- runway
def round1(x: float) -> float:
    """Half-up rounding to 1 decimal (Python's round() is banker's rounding)."""
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


@dataclass
class Shock:
    income_keep: float = 1.0      # r: share of usual money in that still arrives
    bills_up: float = 0.0         # h: household bills increase as a fraction of H
    one_off: float = 0.0          # C: one-off cost taken from the buffer
    cut: float = 0.0              # c: customer-confirmed monthly spending cut (plan only)
    extra_income: float = 0.0     # extra monthly money in the customer confirms (plan only)
    extra_out: float = 0.0        # new monthly costs from the shock (childcare, higher payment, price rises)


@dataclass
class RunwayResult:
    status: str          # "buffer_gone" | "no_shortfall" | "capped" | "shortfall"
    raw: float | None    # unrounded months (None when no shortfall)
    months: float        # value used in the judging formula (0, capped, or raw)
    display: str
    income: float
    spending: float
    gap: float
    buffer_left: float


def runway(B: float, I: float, S: float, H: float, shock: Shock = Shock()) -> RunwayResult:
    income = shock.income_keep * I + shock.extra_income
    spending = S + shock.bills_up * H + shock.extra_out - shock.cut
    buffer_left = B - shock.one_off
    gap = spending - income
    if buffer_left <= 0:
        return RunwayResult("buffer_gone", 0.0, 0.0, "0.0", income, spending, gap, buffer_left)
    if gap <= 0:
        return RunwayResult("no_shortfall", None, RUNWAY_CAP, "no shortfall (120+)",
                            income, spending, gap, buffer_left)
    raw = buffer_left / gap
    if raw >= RUNWAY_CAP:
        return RunwayResult("capped", raw, RUNWAY_CAP, "120.0", income, spending, gap, buffer_left)
    return RunwayResult("shortfall", raw, raw, f"{round1(raw):.1f}", income, spending, gap, buffer_left)


def runway_vectorized(prof: pd.DataFrame, r=1.0, h=0.0, C=0.0, I="I", S="S", B="B") -> pd.Series:
    """Raw (unrounded) judging runway for every account; inf = no shortfall."""
    income = r * prof[I]
    spending = prof[S] + h * prof["H"]
    left = prof[B] - C
    gap = spending - income
    out = pd.Series(np.where(gap > 0, left / gap.where(gap > 0), np.inf), index=prof.index)
    out[left <= 0] = 0.0
    return out


# ---------------------------------------------------------------- plans
def improvement_needed(gap: float, buffer_left: float, target_months: float) -> float | None:
    """Smallest constant monthly improvement so the buffer lasts target_months.

    Returns None when the buffer is already gone (the formula can't fix that).
    """
    if target_months is None or target_months <= 0:
        raise ValueError("Target months must be greater than 0.")
    if buffer_left <= 0:
        return None
    return max(gap - buffer_left / target_months, 0.0)


@dataclass
class FundPlan:
    target: float
    to_build: float
    months: int | None   # None = can't reach with this saving amount
    reached: bool


def emergency_fund(S: float, B: float, k_months: float, monthly_saving: float) -> FundPlan:
    target = k_months * S
    to_build = max(target - B, 0.0)
    if to_build == 0:
        return FundPlan(target, 0.0, 0, True)
    if monthly_saving <= 0:
        return FundPlan(target, to_build, None, False)
    return FundPlan(target, to_build, math.ceil(to_build / monthly_saving), False)


def applied_cut(cut: float, spending_after_shock: float) -> float:
    """Confirmed cut actually applied: never negative, never more than all money out."""
    return float(min(max(cut, 0.0), max(spending_after_shock, 0.0)))


def projection(start: float, gap: float, months: int) -> np.ndarray:
    """Cash if the same monthly money in/out repeats: start − gap × t, t = 0..months.

    Surpluses (gap < 0) grow the line, negative starts stay negative. The judging
    runway is the month where this line first reaches 0.
    """
    t = np.arange(0, months + 1)
    return start - gap * t


def backup_plan(after: RunwayResult, planned: RunwayResult, need: float | None, confirmed: float,
                target: int, loan_payment: float, one_off: float, fmt_money, fmt_runway) -> list[str]:
    """Customer steps for when the shock hits, built from this account's numbers.

    Never reports an open shortfall as solved: whatever the confirmed changes
    don't cover is stated as still open.
    """
    m, steps = fmt_money, []
    if after.status == "buffer_gone":
        if one_off > 0:
            steps.append(f"After the one-off cost of {m(one_off)}, the cushion is {m(after.buffer_left)}. "
                         "Arrange money for this first.")
        else:
            steps.append(f"There's no cushion to draw on (recorded cushion: {m(after.buffer_left)}). "
                         "Arrange money for this first.")
    elif after.status == "no_shortfall":
        steps.append("Money in still covers money out under this shock. Keep the cushion for one-off costs.")
    else:
        steps.append(f"The cushion of {m(after.buffer_left)} covers a {m(after.gap)}/month shortfall for "
                     f"{fmt_runway(after).lower()}.")
    if after.gap > 0:
        if confirmed <= 0:
            steps.append(f"You haven't confirmed any cuts or extra money in yet, so the {m(after.gap)}/month "
                         "shortfall is still open.")
        elif planned.gap <= 0:
            steps.append(f"Your confirmed changes ({m(confirmed)}/month) cover the monthly shortfall.")
        else:
            txt = (f"Your confirmed changes ({m(confirmed)}/month) bring the shortfall down to "
                   f"{m(planned.gap)}/month")
            if planned.status in ("shortfall", "capped"):
                txt += f", so the cushion lasts {fmt_runway(planned).lower()}"
            steps.append(txt + ".")
        if need is not None and need > confirmed:
            line = (f"To last {target} months, you still need another {m(need - confirmed)}/month "
                    "(more cuts, more money in, or outside help).")
            if planned.status == "shortfall":
                line += f" Without it, plan to need help by month {int(planned.raw) + 1}."
            steps.append(line)
        if need is None and planned.gap > 0:
            steps.append(f"Even once that's covered, money out is still {m(planned.gap)}/month more than money in.")
    if loan_payment > 0:
        steps.append(f"The loan payment of {m(loan_payment)}/month is part of money out. "
                     "Ask the lender early about payment options.")
    return steps


# ================================================================ analysis only
# Peer comparison and owner age are no longer shown in the app; analysis/findings.py
# and the tests use them.
# ---------------------------------------------------------------- peers
PEER_METRICS = {
    "S": "Monthly spending",
    "S_over_I": "Spending as a share of money received",
    "F_over_I": "Recorded regular payments (share of money received)",
    "P_over_I": "Current loan payment burden (share of money received)",
    "B_over_S": "Cash cushion relative to spending (months)",
}

# ---------------------------------------------------------------- owner age
AGE_REF_DATE = pd.Timestamp("2025-12-31")
AGE_STATUS_TEXT = {
    "ok": "",
    "no_owner": "no account owner is recorded",
    "multiple_owners": "more than one owner is recorded",
    "missing_client": "the owner's record is missing",
    "bad_birth_date": "the owner's birth date isn't valid",
    "born_after_ref": "the owner's birth date is after the reference date",
}


def age_on(birth: pd.Timestamp, ref: pd.Timestamp = AGE_REF_DATE) -> int:
    """Completed years at ref (accounts for whether the birthday has passed)."""
    return ref.year - birth.year - int((ref.month, ref.day) < (birth.month, birth.day))


def owner_ages(dispositions: pd.DataFrame, clients: pd.DataFrame, account_ids,
               ref: pd.Timestamp = AGE_REF_DATE) -> pd.DataFrame:
    """One row per account: owner_age (NaN if unusable) and age_status.

    Never picks 'the first owner' — accounts with 0 or 2+ owners, a missing client,
    or a bad / future birth date get a status and no age.
    """
    own = dispositions[dispositions["type"] == "owner"]
    n_owners = own.groupby("account_id").size()
    out = pd.DataFrame(index=pd.Index(account_ids, name="account_id"))
    out["n_owners"] = n_owners.reindex(out.index).fillna(0).astype(int)
    single = own[own["account_id"].map(n_owners) == 1].set_index("account_id")["client_id"]
    cl = clients.drop_duplicates("client_id", keep=False).set_index("client_id")["birth_date"]
    birth_raw = single.map(cl)
    birth = pd.to_datetime(birth_raw, format="%Y-%m-%d", errors="coerce")   # e.g. 1980-02-30 -> invalid
    out["status"] = "ok"
    out.loc[out["n_owners"] == 0, "status"] = "no_owner"
    out.loc[out["n_owners"] > 1, "status"] = "multiple_owners"
    is_single = out["n_owners"] == 1
    missing = is_single & birth_raw.reindex(out.index).isna()
    bad = is_single & ~missing & birth.reindex(out.index).isna()
    future = is_single & (birth.reindex(out.index) > ref)
    out.loc[missing, "status"] = "missing_client"
    out.loc[bad, "status"] = "bad_birth_date"
    out.loc[future, "status"] = "born_after_ref"
    ok = out["status"] == "ok"
    out["owner_age"] = np.nan
    out.loc[ok, "owner_age"] = [age_on(b, ref) for b in birth.reindex(out.index)[ok]]
    return out[["owner_age", "status"]].rename(columns={"status": "age_status"})


def add_peer_metrics(prof: pd.DataFrame) -> pd.DataFrame:
    p = prof.copy()
    pos_i = p["I"] > 0
    p["S_over_I"] = (p["S"] / p["I"]).where(pos_i)
    p["F_over_I"] = (p["F"] / p["I"]).where(pos_i)
    p["P_over_I"] = (p["P"] / p["I"]).where(pos_i)
    p["B_over_S"] = (p["B"] / p["S"]).where(p["S"] > 0)
    return p


NOT_ENOUGH_AGE = "Not enough comparable accounts in this age range. Try comparing by income only."


NOT_ENOUGH_AGE_ONLY = "Not enough comparable accounts in this age range."


def peer_group(prof: pd.DataFrame, account_id: int, band: float = 0.20,
               min_n: int = 30, widen: tuple = (0.30, 0.50), age_band: int | None = None,
               use_income: bool = True) -> dict:
    """Peer accounts for one account (the account itself is always excluded).

    use_income=True: typical money in within ±band, widened step by step if fewer
    than min_n accounts match. age_band: owners must also be within ±age_band years
    (age at AGE_REF_DATE); the age condition is never dropped while widening.
    use_income=False (needs age_band): age only, no widening.
    """
    if not use_income and age_band is None:
        raise ValueError("An age-only comparison needs age_band.")
    I0 = prof.at[account_id, "I"]
    if use_income and not I0 > 0:
        return {"ok": False, "reason": "This account has no typical money in to compare against."}
    others = prof.drop(index=account_id)
    age_lo = age_hi = None
    if age_band is not None:
        status = prof.at[account_id, "age_status"]
        if status != "ok":
            return {"ok": False, "age_unavailable": True,
                    "reason": f"We can't compare by age for this account because {AGE_STATUS_TEXT[status]}. "
                              "Comparing by income only still works."}
        a0 = prof.at[account_id, "owner_age"]
        age_lo, age_hi = a0 - age_band, a0 + age_band
        others = others[(others["age_status"] == "ok") & (others["owner_age"] >= age_lo)
                        & (others["owner_age"] <= age_hi)]
    if not use_income:
        if len(others) >= min_n:
            return {"ok": True, "band": None, "lo": None, "hi": None, "group": others, "widened": False,
                    "age_lo": age_lo, "age_hi": age_hi}
        return {"ok": False, "reason": NOT_ENOUGH_AGE_ONLY, "tried": [(None, len(others))],
                "age_lo": age_lo, "age_hi": age_hi}
    tried = []
    for b in (band, *widen):
        lo, hi = I0 * (1 - b), I0 * (1 + b)
        grp = others[(others["I"] >= lo) & (others["I"] <= hi)]
        tried.append((b, len(grp)))
        if len(grp) >= min_n:
            return {"ok": True, "band": b, "lo": lo, "hi": hi, "group": grp, "widened": b != band,
                    "age_lo": age_lo, "age_hi": age_hi}
    reason = NOT_ENOUGH_AGE if age_band is not None else "Not enough similar accounts to compare."
    return {"ok": False, "reason": reason, "tried": tried, "age_lo": age_lo, "age_hi": age_hi}


def peer_stats(grp: pd.DataFrame, metric: str) -> dict:
    s = grp[metric].dropna()
    return {"n": int(len(s)), "median": s.median(), "p25": s.quantile(0.25),
            "p75": s.quantile(0.75), "mean": s.mean()}


def rank_position(grp: pd.DataFrame, metric: str, value: float) -> dict | None:
    """Share of peers strictly below / exactly equal to this value (missing values ignored)."""
    s = grp[metric].dropna()
    if len(s) == 0 or value is None or pd.isna(value):
        return None
    return {"lower_pct": float((s < value).mean() * 100),
            "equal_pct": float((s == value).mean() * 100), "n": int(len(s))}


def rank_text(r: dict | None) -> str:
    """'higher than X%' uses only strictly lower peers; ties are reported separately."""
    if r is None:
        return "—"
    txt = f"higher than {r['lower_pct']:.0f}%"
    if r["equal_pct"] > 0:
        txt += f" · same as {r['equal_pct']:.0f}%"
    return txt
