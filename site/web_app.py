"""Resilience Rewired — standalone web app (JSON API + single-page frontend).

Every number comes from engine.py (judging formulas), response_plan.py (response state)
and planning_guide.py (guide wording). This file only loads data, routes requests and
serialises results; it never re-derives a financial figure.

Run:   .venv/bin/python web_app.py            → http://127.0.0.1:8600
       .venv/bin/python web_app.py --port 9000

Endpoints
  GET  /api/overview        account count, period + every account (for type-ahead)
  GET  /api/account/<id>    profile, observed period, money out for the last 12 recorded months, peer benchmark
  POST /api/simulate        shock + plan → runways, projection, plan, guide, backup,
                            ranked data-driven plans, risk, month-by-month, message
  POST /api/twist           twist-card text → shock settings (Gemini, rule-based backup)
  POST /api/message         same body as /api/simulate → customer message rewritten by Gemini
  GET  /api/findings        the six findings (past features → what happened next)

Scope: the 3-tab submission app plus the peer benchmark (income ±20%, owner age ±5, both).
No daily activity, findings or random account; those live in analysis/ (see analysis/FINDINGS.md).
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT_REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_REPO))               # core.py + agent.py live in the repo root (site/engine.py stays first)


def _load_secrets():
    """Gemini key from the Streamlit secrets file, so both front ends share one key."""
    import os
    f = ROOT_REPO / ".streamlit" / "secrets.toml"
    if f.exists() and "GEMINI_API_KEY" not in os.environ:
        import tomllib
        s = tomllib.load(open(f, "rb"))
        for k in ("GEMINI_API_KEY", "GEMINI_MODEL"):
            if s.get(k):
                os.environ[k] = s[k]


_load_secrets()
import agent  # noqa: E402
import core  # noqa: E402
import engine as E  # noqa: E402
import planning_guide as PG
import response_plan as RP

ROOT = Path(__file__).parent
WEB = ROOT / "web"


# ================================================================ formatting (same as the Streamlit app)
def money(x, signed=False, cents=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    fmt = "{:,.2f}" if cents else "{:,.0f}"
    s = "$" + fmt.format(abs(x))
    if x < 0:
        return "−" + s
    return ("+" + s) if signed and x > 0 else s


def runway_text(r: E.RunwayResult) -> tuple[str, str]:
    if r.status == "buffer_gone":
        return "0 months", "There's no cushion left to draw on."
    if r.status == "no_shortfall":
        return "No shortfall", "Money in still covers money out each month."
    if r.status == "capped":
        return "120+ months", f"Calculated as {r.raw:,.0f} months; we cap it at 120."
    return f"{r.display} months", f"Short {money(r.gap)} a month, drawn from a {money(r.buffer_left)} cushion."


def num(x):
    """JSON-safe float (NaN/inf → None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def runway_json(r: E.RunwayResult) -> dict:
    h, sub = runway_text(r)
    return {"status": r.status, "raw": num(r.raw), "display": r.display, "headline": h, "sub": sub,
            "income": num(r.income), "spending": num(r.spending), "gap": num(r.gap),
            "buffer_left": num(r.buffer_left)}


def fund_json(f: E.FundPlan) -> dict:
    return {"target": num(f.target), "to_build": num(f.to_build), "months": f.months, "reached": f.reached}


# ================================================================ data (loaded once)
class Store:
    def __init__(self):
        t = time.time()
        self.ms = E.load_monthly()
        try:                                       # raw CSVs present (data/ or site/data/)
            loans, disp, clients = E.load_loans(), E.load_dispositions(), E.load_clients()
            prof = E.add_peer_metrics(E.compute_profiles(self.ms, loans))
            ages = E.owner_ages(disp, clients, prof.index)
            src = "raw CSVs in " + str(E.DATA_DIR)
        except FileNotFoundError:                  # only the repo's app_data/: P and owner ages were saved there
            import pandas as pd
            extra = pd.read_csv(E.DATA_DIR / "site_extra.csv", index_col="account_id")
            run = extra[extra.P > 0]
            loans = pd.DataFrame({"account_id": run.index, "monthly_payment": run.P.values, "status": "C"})
            prof = E.add_peer_metrics(E.compute_profiles(self.ms, loans))
            ages = extra[["owner_age", "age_status"]]
            src = "app_data/ (no raw CSVs needed)"
        self.prof = prof.join(ages, how="left")
        assert len(self.prof) == len(prof)                       # the age join must not add rows
        totals = self.ms.groupby("account_id")[list(E.OUT_COMPONENTS)].sum()
        self.shares = totals.div(totals.sum(axis=1).replace(0, np.nan), axis=0)
        self.invalid = RP.invalid_accounts(self.ms)
        self.by_acct = {a: g.reset_index(drop=True) for a, g in self.ms.groupby("account_id", sort=False)}
        self.overview = self._overview()
        self.findings = core.findings_data()
        print(f"data ready in {time.time() - t:.1f}s · {len(self.prof):,} accounts · from {src}", flush=True)

    # ------------------------------------------------------------ landing
    def _overview(self) -> dict:
        p, ms = self.prof, self.ms
        accounts = []
        for aid, row in p.iterrows():
            r = E.runway(row.B, row.I, row.S, row.H)
            accounts.append({"id": int(aid), "B": round(float(row.B), 2), "runway": runway_text(r)[0],
                             "status": r.status})
        return {
            "n_accounts": int(len(p)),
            "period": [ms["month"].min(), ms["month"].max()],
            "accounts": accounts,
        }

    # ------------------------------------------------------------ one account
    def account(self, aid: int) -> dict:
        p = self.prof.loc[aid]
        m = self.by_acct[aid].iloc[-12:]
        return {
            "id": aid,
            "profile": {k: num(p[k]) for k in ("I", "S", "F", "H", "B", "P")},
            "first_month": p.first_month, "last_month": p.last_month, "n_months": int(p.n_months),
            "data_ok": aid not in self.invalid,
            "base": runway_json(E.runway(p.B, p.I, p.S, p.H)),
            "spending": [{"month": r["month"], "money_out": num(r["money_out"])} for r in m.to_dict("records")],
            "peers": self.peers(aid),
        }

    def peers(self, aid: int) -> dict:
        """Peer benchmark: a comparison, never a spending target or a cuttable amount."""
        prof, p = self.prof, self.prof.loc[aid]
        modes = {"income": dict(), "age": dict(age_band=5, use_income=False), "both": dict(age_band=5)}
        out = {}
        for name, kw in modes.items():
            g = E.peer_group(prof, aid, **kw)
            if not g["ok"]:
                out[name] = {"ok": False, "reason": g["reason"]}
                continue
            grp = g["group"]
            rows = []
            for key, label in E.PEER_METRICS.items():
                st = E.peer_stats(grp, key)
                rows.append({"key": key, "label": label, "you": num(p[key]), "median": num(st["median"]),
                             "p25": num(st["p25"]), "p75": num(st["p75"]), "n": st["n"],
                             "position": E.rank_text(E.rank_position(grp, key, p[key]))})
            mix_you, mix_peer = self.shares.loc[aid], self.shares.loc[grp.index].median()
            out[name] = {"ok": True, "n": int(len(grp)), "band": num(g.get("band")), "widened": bool(g["widened"]),
                         "lo": num(g.get("lo")), "hi": num(g.get("hi")),
                         "age_lo": num(g.get("age_lo")), "age_hi": num(g.get("age_hi")), "rows": rows,
                         "mix": [{"key": k, "label": E.OUT_COMPONENTS[k], "you": num(mix_you[k]),
                                  "peer": num(mix_peer[k])} for k in E.OUT_COMPONENTS]}
        return out

    # ------------------------------------------------------------ shock + plan
    def scenario(self, q: dict) -> dict:
        return scenario_of(q, float(self.prof.loc[int(q["account_id"])].S))

    def twist(self, q: dict) -> dict:
        """Twist card text → the same controls a person would set by hand."""
        text = str(q.get("text", "")).strip()[:2000]
        if not text:
            raise ValueError("Paste a twist card first.")
        t, src = agent.parse_twist(text)
        p = self.prof.loc[int(q["account_id"])]
        ei = float(t["ei_pct"] or 0) if t["job_loss"] else 100.0
        keep = ei * (1 + float(t["other_income_change_pct"] or 0) / 100)
        one_off = float(t["one_off_dollars"] or 0) + float(t["one_off_months_of_spending"] or 0) * p.S
        extra_out = float(t["extra_monthly_dollars"] or 0) + float(t["spending_change_pct"] or 0) / 100 * p.S
        warn = list(t.get("unsupported") or [])
        bills = float(t["bills_up_pct"] or 0)
        if bills > 500:
            warn.append(f"bills +{bills:g}% capped at +500%")
        note = f"Read by {'Gemini (' + agent.USED_MODEL + ')' if src == 'Gemini' else src}: {agent.describe(t)}"
        if src == "Gemini" and agent.ADDED:
            note += f" (rule check added: {'; '.join(agent.ADDED)})"
        if src != "Gemini" and agent.LAST_ERROR:
            note += f" · backup reader used ({agent.short_error()})"
        parts = []
        if t["job_loss"]:
            parts.append(f"job loss (EI {ei:g}%)")
        if t["other_income_change_pct"]:
            parts.append(f"income {float(t['other_income_change_pct']):+g}%")
        return {"source": src, "note": note, "warn": warn, "label": ", ".join(parts),
                "shock": {"keep": round(min(max(keep, 0), 150), 1), "bills": round(min(bills, 500), 1),
                          "oneOff": round(one_off), "extraOut": round(extra_out),
                          "windfall": round(float(t.get("windfall_dollars") or 0)),
                          "startIn": int(t.get("starts_in_months") or 0), "lasts": int(t.get("shock_lasts_months") or 0),
                          "afterPct": round(float(t.get("income_after_pct", 100)), 1)},
                "question": t.get("question", "auto"),
                "target": int(min(max(int(t["survive_months"]), 1), 24)) if t.get("survive_months") else None}

    def message(self, q: dict) -> dict:
        _, facts = self._simulate(q)          # this request's own facts: nothing shared with other requests
        msg, src = agent.write_message(facts)
        return {"message": msg, "source": src, "error": "" if src == "Gemini" else agent.short_error()}

    def simulate(self, q: dict) -> dict:
        return self._simulate(q)[0]

    def _simulate(self, q: dict) -> tuple[dict, dict]:
        """(API result, the facts the customer message is written from) for one request."""
        aid = int(q["account_id"])
        p = self.prof.loc[aid]
        sc = self.scenario(q)
        keep, bills, one_off, extra_out, windfall = sc["keep"], sc["bills"], sc["one_off"], sc["extra_out"], sc["windfall"]
        net = one_off - windfall                 # what actually leaves the cushion (negative = money in)
        target = int(min(max(int(q.get("target", 6)), 1), 60))
        checked = bool(q.get("checked", False))
        entered_cut = max(float(q.get("cut", 0)), 0.0) if checked else 0.0
        extra = max(float(q.get("extra", 0)), 0.0)
        kmon = int(q.get("kmon", 3))
        save = max(float(q.get("save", 0)), 0.0)   # monthly saving for the fund timeline only, never cash today

        base = E.runway(p.B, p.I, p.S, p.H)
        after = E.runway(p.B, p.I, p.S, p.H, E.Shock(keep, bills, net, extra_out=extra_out))
        cut = E.applied_cut(entered_cut, after.spending)
        # the plan changes the monthly flow only (confirmed cut / extra money in); the starting cash is the same
        # as after the shock. Savings goals are future money and are never added here.
        planned = E.runway(p.B, p.I, p.S, p.H, E.Shock(keep, bills, net, cut=cut, extra_income=extra,
                                                         extra_out=extra_out))
        need = E.improvement_needed(after.gap, after.buffer_left, target)
        confirmed = cut + extra
        remaining = None if need is None else max(need - confirmed, 0.0)
        lasts = after.status in ("no_shortfall", "capped") or (after.status == "shortfall" and after.raw >= target)
        short_by = max(target * max(after.gap, 0) - max(after.buffer_left, 0), 0.0)

        horizon = 12
        for r in (after, planned):
            if r.status == "shortfall" and r.raw:
                horizon = max(horizon, min(math.ceil(r.raw * 1.25), 60))
        fund = E.emergency_fund(p.S, p.B, kmon, save)
        fund_after = E.emergency_fund(p.S, after.buffer_left, kmon, save)
        response = RP.classify(float(after.buffer_left), float(after.gap), float(target))
        if aid in self.invalid:
            response = RP.classify(float("nan"), float(after.gap), float(target))
        guide = PG.build(response, after, planned, target, max(net, 0.0), confirmed, extra, fund, save, kmon, money,
                         fund_after=fund_after)
        backup = E.backup_plan(after, planned, need, confirmed, target, p.P, max(net, 0.0),
                               fmt_money=money, fmt_runway=lambda r: runway_text(r)[0])
        # ---- month by month (timing), can-they-pay, ranked plans, message: core.py + agent.py
        timed = None
        if sc["start_in"] or sc["lasts"]:
            out, path = core.simulate(p.B, p.I, p.S, after.income, after.spending, net,
                                      start_in=sc["start_in"], lasts=sc["lasts"], after_frac=sc["after"])
            timed = {"out": num(out), "start_in": sc["start_in"], "lasts": sc["lasts"], "after_pct": sc["after"] * 100}
            if out is not None:
                horizon = max(horizon, min(math.ceil(out * 1.25), 60))
            else:
                horizon = max(horizon, min(sc["start_in"] + sc["lasts"] + 6, 60))
            timed["path"] = [num(v) for v in path[:horizon + 1]]
        one_off_only = keep >= 1 and bills == 0 and extra_out == 0 and one_off > 0
        question = sc["question"] if sc["question"] != "auto" else ("can_pay" if one_off_only else "last_months")
        if question == "can_pay" and one_off <= 0:
            question = "last_months"
        can_pay = None
        if question == "can_pay":
            # Paying and having cash left afterwards are separate answers: cost == savings can be paid,
            # with nothing left (the judges' runway is then 0, unchanged). Decided on unrounded values.
            have = float(p.B + windfall)
            left = have - float(one_off)
            can_pay = {"cost": num(one_off), "have": num(have), "left": num(left), "windfall": num(windfall),
                       "can_pay": left >= 0, "cash_left": left > 0}
        plans, risk, _ = core.recommend(aid, dict(I=p.I, S=p.S, F=p.F, B=p.B, H=p.H), keep, bills, net,
                                        target, min(kmon, 6), extra_out)
        facts = dict(customer=aid, shock=sc["text"], runway_now=core.runway(p.B, p.I, p.S),
                     runway_after=core.runway(p.B, keep * p.I, p.S + bills * p.H + extra_out, net),
                     survive=target, one_off_only=question == "can_pay", one_off=round(one_off),
                     buffer=round(p.B + windfall),
                     can_pay=None if can_pay is None else can_pay["can_pay"],
                     left=None if can_pay is None else can_pay["left"],
                     month_by_month=None if timed is None else (round(timed["out"], 1) if timed["out"] is not None
                                                                else "does not run out"),
                     steps=[(pl["do"], pl["bank"]) for pl in plans])
        projection_extra = {"timed": timed["path"] if timed else None}
        return {
            "question": question, "can_pay": can_pay, "timed": timed, "shock_text": sc["text"],
            "risk": {"base": num(risk), "auc": num(core.AUC)},
            "plans": [plan_json(pl) for pl in plans],
            "message": agent.template_message(facts),
            "account_id": aid, "target": target, "plan_active": confirmed > 0,
            "base": runway_json(base), "after": runway_json(after), "planned": runway_json(planned),
            "lasts": lasts, "short_by": num(short_by),
            "need": num(need), "confirmed": num(confirmed), "applied_cut": num(cut),
            "entered_cut": num(entered_cut), "cut_clamped": entered_cut > cut, "extra": num(extra),
            "remaining": num(remaining), "one_off": num(one_off),
            "projection": {"months": list(range(horizon + 1)),
                           "today": [num(v) for v in E.projection(base.buffer_left, base.gap, horizon)],
                           "shock": [num(v) for v in E.projection(after.buffer_left, after.gap, horizon)],
                           "plan": [num(v) for v in E.projection(planned.buffer_left, planned.gap, horizon)],
                           **projection_extra},
            "fund": fund_json(fund), "fund_after": fund_json(fund_after), "kmon": kmon, "save": num(save),
            "response": {"group": response.group, "label": RP.LABELS[response.group]},
            "guide": {"why": guide.why, "actions": guide.actions, "result": guide.result, "open_gap": guide.open_gap},
            "backup": backup,
        }, facts


def plan_json(pl: dict) -> dict:
    text = pl["pill"][0]
    kind = ("safety" if pl.get("pinned") else "support" if pl.get("infeasible")
            else "rank" if pl.get("sig") else "low")
    return {"pill": text, "kind": kind, "title": pl["title"], "do": pl["do"], "you": pl["you"], "you_c": pl["you_c"],
            "data": pl["data"], "bank": pl["bank"], "start": num(pl.get("start")),
            "new": num(pl["new"]) if pl["new"] is not None else None,
            "red": num(pl["red"]) if pl["red"] is not None else None, "test": pl.get("test", ""), "track": pl.get("track", ""),
            "apply": None if not pl.get("apply") else {k: (num(v) if k != "label" else v) for k, v in pl["apply"].items()},
            "goal": num(pl.get("goal")), "goal_note": goal_note(pl.get("goal")),
            "sig": bool(pl.get("sig"))}


def goal_note(goal) -> str:
    """Savings goals are money still to be saved: say so, and never present the model's 'after' risk as a
    change that has already happened."""
    if not goal:
        return ""
    return (f"Savings goal ({money(goal)}): this money isn't saved yet, so it isn't added to today's cash or the "
            f"runway chart. The risk figures compare customers who already have it saved; nothing changes "
            f"until the goal is reached.")


def _f(q, k, default, lo, hi):
    return float(min(max(float(q.get(k, default) or 0), lo), hi))


def scenario_of(q: dict, S_typ: float) -> dict:
    """Shock inputs from a request, clamped. keep = share of usual money in that continues (EI x other change)."""
    sc = {
        "keep": _f(q, "keep", 100, 0, 150) / 100,
        "bills": _f(q, "bills_pct", 0, 0, 500) / 100,
        "one_off": _f(q, "one_off", 0, 0, 1e7),
        "extra_out": _f(q, "extra_out", 0, -S_typ, 1e6),
        "windfall": _f(q, "windfall", 0, 0, 1e7),
        "start_in": int(_f(q, "start_in", 0, 0, 24)),
        "lasts": int(_f(q, "lasts", 0, 0, 60)),
        "after": _f(q, "after_pct", 100, 0, 150) / 100,
        "question": q.get("question", "auto") if q.get("question") in ("auto", "can_pay", "last_months", "how_long") else "auto",
    }
    d = []
    label = str(q.get("label") or "").strip()[:80]
    if label:
        d.append(label)
    elif sc["keep"] < 1:
        d.append(f"income at {sc['keep'] * 100:g}% of usual")
    elif sc["keep"] > 1:
        d.append(f"income at {sc['keep'] * 100:g}% of usual")
    if sc["bills"]:
        d.append(f"bills +{sc['bills'] * 100:g}%")
    if sc["extra_out"]:
        d.append(f"new costs {money(sc['extra_out'])}/month" if sc["extra_out"] > 0 else
                 f"spending down {money(-sc['extra_out'])}/month")
    if sc["one_off"]:
        d.append(f"one-off cost {money(sc['one_off'])}")
    if sc["windfall"]:
        d.append(f"one-off money in {money(sc['windfall'])}")
    if sc["start_in"]:
        d.append(f"starts in {sc['start_in']} months")
    if sc["lasts"]:
        d.append(f"lasts {sc['lasts']} months")
    sc["text"] = ", ".join(d) or "no shock"
    return sc


# ================================================================ request validation
class BadRequest(ValueError):
    def __init__(self, field: str | None, msg: str):
        super().__init__(msg)
        self.field = field


# field -> (lowest, highest, whole number only). Ranges are the ones the engine already clamps to (or wider
# where the page can send more, e.g. a typed negative "new monthly cost"); anything outside is a 400.
NUMBER_FIELDS = {
    "keep": (0, 150, False), "bills_pct": (0, 500, False), "one_off": (0, 1e7, False),
    "extra_out": (-1e9, 1e6, False), "windfall": (0, 1e7, False),
    "start_in": (0, 24, True), "lasts": (0, 60, True), "after_pct": (0, 150, False),
    "target": (1, 60, True), "cut": (0, 1e9, False), "extra": (0, 1e9, False),
    "kmon": (1, 24, True), "save": (0, 1e9, False), "plan_buffer": (0, 1e8, False),
}
BOOL_FIELDS = ("checked",)
TEXT_FIELDS = ("question", "label", "text")


def validate_request(path: str, q) -> dict:
    """Checks the JSON body of a POST before any calculation. Raises BadRequest (-> 400) on a wrong type,
    NaN/Infinity, an out-of-range value or a missing account_id. A null number means 'use the default'.
    Whether the account exists is checked afterwards (-> 404)."""
    if not isinstance(q, dict):
        raise BadRequest(None, "The request body must be a JSON object.")
    aid = q.get("account_id")
    if aid is None:
        raise BadRequest("account_id", "account_id is required.")
    if isinstance(aid, bool) or not (isinstance(aid, int) or (isinstance(aid, str) and aid.strip().isdigit())):
        raise BadRequest("account_id", "account_id must be a whole number.")
    out = dict(q)
    for k, (lo, hi, whole) in NUMBER_FIELDS.items():
        if k not in q:
            continue
        v = q[k]
        if v is None:
            out.pop(k)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):       # bool is an int in Python
            raise BadRequest(k, f"{k} must be a number.")
        if isinstance(v, float) and not math.isfinite(v):   # ints are exact (any size): no float conversion
            raise BadRequest(k, f"{k} must be a finite number.")
        if not lo <= v <= hi:                               # int vs float compares exactly, even 10**400
            raise BadRequest(k, f"{k} must be between {lo:g} and {hi:g}.")
        if whole and v != int(v):
            raise BadRequest(k, f"{k} must be a whole number.")
    for k in BOOL_FIELDS:
        if k in q and q[k] is not None and not isinstance(q[k], bool):
            raise BadRequest(k, f"{k} must be true or false.")
    for k in TEXT_FIELDS:                       # length is trimmed later by the route, as before
        if k in q and q[k] is not None and not isinstance(q[k], str):
            raise BadRequest(k, f"{k} must be text.")
    if path == "/api/twist" and not isinstance(q.get("text"), str):
        raise BadRequest("text", "text is required.")
    if out.get("plan_buffer"):          # range-checked above; 0 / null / missing from older pages are fine
        raise BadRequest("plan_buffer", "plan_buffer is no longer supported: savings goals are money still to be "
                                        "saved, so they can't be added to today's cash. Send 0 or leave it out.")
    out.pop("plan_buffer", None)        # not used by any calculation
    return out


STORE: Store | None = None


def _jsonable(o):
    """numpy scalars → plain Python (pandas/numpy comparisons return np.bool_, np.float64, …)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return num(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_jsonable)


# ================================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ResilienceRewired/1.0"

    def log_message(self, fmt, *args):   # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # fonts never change; html/css/js revalidate so an edit shows on the next reload
        self.send_header("Cache-Control", "max-age=86400" if ctype.startswith("font/") else "no-cache")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass        # the page cancelled this request (a newer slider value is on its way): nothing to send

    def _json(self, obj, code=200):
        self._send(code, dumps(obj).encode(), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _account_id(self, raw: str):
        try:
            aid = int(raw)
        except ValueError:
            return None
        return aid if aid in STORE.prof.index else None

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/overview":
                return self._json(STORE.overview)
            if path == "/api/findings":
                return self._json(STORE.findings)
            parts = path.strip("/").split("/")
            if parts[:2] == ["api", "account"] and len(parts) >= 3:
                aid = self._account_id(parts[2])
                if aid is None:
                    return self._error(404, f"We couldn't find account {parts[2]}.")
                if len(parts) == 3:
                    return self._json(STORE.account(aid))
                return self._error(404, "Not found.")
            return self._static(path)
        except Exception as exc:  # pragma: no cover - surfaced to the client
            return self._error(500, f"Server error: {exc}")

    def do_POST(self):
        path = urlparse(self.path).path
        routes = {"/api/simulate": STORE.simulate, "/api/twist": STORE.twist, "/api/message": STORE.message}
        if path not in routes:
            return self._error(404, "Not found.")
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n > 0 else b""
            try:
                q = json.loads(raw or b"{}")
            except ValueError as exc:                     # includes bad UTF-8
                raise BadRequest(None, f"The request body is not valid JSON ({exc}).") from None
            q = validate_request(path, q)
        except BadRequest as exc:
            return self._json({"error": f"Bad request: {exc}", "field": exc.field}, 400)
        except ValueError:
            return self._json({"error": "Bad request: Content-Length must be a whole number.", "field": None}, 400)
        if self._account_id(str(q["account_id"]).strip()) is None:
            return self._error(404, "We couldn't find that account.")
        try:
            return self._json(routes[path](q))
        except ValueError as exc:                         # e.g. an empty twist card
            return self._json({"error": f"Bad request: {exc}", "field": None}, 400)
        except Exception as exc:  # pragma: no cover - a bug, not bad input: answer and keep the traceback
            import traceback
            traceback.print_exc()
            return self._error(500, f"Server error: {type(exc).__name__}")

    def _static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents and f != WEB.resolve() or not f.is_file():
            return self._error(404, "Not found.")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".woff2":
            ctype = "font/woff2"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)


def load_store() -> Store:
    global STORE
    if STORE is None:
        STORE = Store()
    return STORE


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return              # browser closed the connection early: normal while dragging a slider
        super().handle_error(request, client_address)


def make_server(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    load_store()
    return QuietServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Resilience Rewired web app")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    srv = make_server(a.port, a.host)
    print(f"Resilience Rewired → http://{a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
