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
import re
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
        if t.get("extra_income_dollars") and p.I > 0:      # $X more/less money in per month, as a share of usual
            keep += float(t["extra_income_dollars"]) / float(p.I) * 100
        one_off = float(t["one_off_dollars"] or 0) + float(t["one_off_months_of_spending"] or 0) * p.S
        extra_out = float(t["extra_monthly_dollars"] or 0) + float(t["spending_change_pct"] or 0) / 100 * p.S
        warn = list(t.get("unsupported") or [])
        if t.get("extra_income_dollars") and p.I <= 0:
            warn.append("no usual money in to scale: set 'Money in that continues' by hand")
        if (t.get("shock_lasts_months") or 0) > 0 and (t["bills_up_pct"] or t["extra_monthly_dollars"]
                                                        or t["spending_change_pct"]) and \
                (t["job_loss"] or t["other_income_change_pct"]):
            warn.append(f"the month-by-month line ends ALL these changes after {plural(t['shock_lasts_months'], 'month')}; "
                        f"if a cost change is permanent, set 'Shock lasts' to Ongoing")
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
        shock = {"keep": round(min(max(keep, 0), 150), 1), "bills": round(min(bills, 500), 1),
                 "oneOff": round(one_off), "extraOut": round(extra_out),
                 "windfall": round(float(t.get("windfall_dollars") or 0)),
                 "startIn": int(t.get("starts_in_months") or 0), "lasts": int(t.get("shock_lasts_months") or 0),
                 "afterPct": round(float(t.get("income_after_pct", 100)), 1)}
        target = int(min(max(int(t["survive_months"]), 1), 24)) if t.get("survive_months") else None
        return {"source": src, "note": note, "warn": warn, "label": ", ".join(parts), "shock": shock,
                "question": t.get("question", "auto"), "target": target,
                "fields": read_fields(text, t, shock, target, p, warn),
                # what could NOT be put into a number (shown in red, to enter by hand); assumptions are in fields
                "missing": [w for w in warn if not w.startswith("partner")]}

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
        # a preview draws a cut on the chart before the customer confirms it; it never counts as confirmed
        preview_cut = 0.0 if checked else E.applied_cut(max(float(q.get("preview_cut", 0) or 0), 0.0), after.spending)
        preview = None if preview_cut <= 0 else E.runway(p.B, p.I, p.S, p.H, E.Shock(
            keep, bills, net, cut=preview_cut, extra_income=extra, extra_out=extra_out))
        need = E.improvement_needed(after.gap, after.buffer_left, target)
        confirmed = cut + extra
        remaining = None if need is None else max(need - confirmed, 0.0)
        lasts = after.status in ("no_shortfall", "capped") or (after.status == "shortfall" and after.raw >= target)
        # shortfall = how far below $0 the balance gets within the target window (right after the one-off, or at
        # the end of the window): the same definition the month-by-month engine uses
        L = float(after.buffer_left)
        short_by = 0.0 if lasts else max(-min(L, L - float(after.gap) * target), 0.0)
        # A shock with timing (starts later / ends): ONE engine, the month-by-month balance, answers every question
        # below (verdict, shortfall, monthly change needed, plan result, guide, backup, message, chart). With no
        # timing the judges' runway is used exactly as before (the two agree when the change never ends).
        T = None
        if sc["start_in"] or sc["lasts"]:
            T = timed_engine(p, after, net, sc, target, confirmed, preview_cut + extra if preview_cut else 0.0)
            lasts, short_by, need = T["lasts"], T["short_by"], T["need"]
            remaining = None if need is None else max(need - confirmed, 0.0)

        horizon = 12
        for r in (after, planned) + ((preview,) if preview else ()):
            if r.status == "shortfall" and r.raw:
                horizon = max(horizon, min(math.ceil(r.raw * 1.25), 60))
        if T:
            for o in (T["out"], T["plan_out"], T["preview_out"]):
                horizon = max(horizon, min(math.ceil(o * 1.25), 60) if o is not None
                              else min(sc["start_in"] + sc["lasts"] + 6, 60))
        fund = E.emergency_fund(p.S, p.B, kmon, save)
        fund_after = E.emergency_fund(p.S, after.buffer_left, kmon, save)
        response = RP.classify(float(after.buffer_left), float(after.gap), float(target))
        if aid in self.invalid:
            response = RP.classify(float("nan"), float(after.gap), float(target))
        if T:
            response = RP.Response(T["group"], need, None, "", "")
            guide, backup = timed_texts(T, sc, target, confirmed, need, remaining, p.P)
        else:
            guide = PG.build(response, after, planned, target, max(net, 0.0), confirmed, extra, fund, save, kmon,
                             money, fund_after=fund_after)
            backup = E.backup_plan(after, planned, need, confirmed, target, p.P, max(net, 0.0),
                                   fmt_money=money, fmt_runway=lambda r: runway_text(r)[0])
        # ---- month by month (timing), can-they-pay, ranked plans, message: core.py + agent.py
        timed = None
        if T:
            timed = {"out": num(T["out"]), "start_in": sc["start_in"], "lasts": sc["lasts"],
                     "after_pct": sc["after"] * 100, "path": [num(v) for v in T["path"][:horizon + 1]],
                     "plan_out": num(T["plan_out"]), "plan_lasts": T["plan_lasts"],
                     "preview_out": num(T["preview_out"]), "need": num(need)}
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
        if T:
            # the shock plan is rewritten from the same month-by-month engine, so its title, steps and the cut it
            # puts on the chart all agree with the verdict and the guide
            flex = max(float(p.S - p.F), 0.0)
            rest = (f"the cushion does not run out" if T["out"] is None
                    else f"the cushion runs out in about {T['out']:.1f} months")
            for pl in plans:
                if not str(pl.get("title", "")).startswith("Shock plan"):
                    continue
                pl["you"] = (f"Counting month by month with the timing you set, {rest} (target: {target} months). "
                             f"Spending outside fixed costs is ${flex:,.0f}/month; not all of it can be cut, so the "
                             f"plan counts only cuts the customer confirms.")
                pl["track"] = ""
                if need is not None and need <= 0:
                    pl["title"] = f"Shock plan: no cut needed to last {target} months with this timing"
                    pl["do"] = pl["you_c"] = (f"Month by month, {rest}, which covers the {target}-month target. "
                                              "Keep the cushion for surprises.")
                    pl["apply"], pl["infeasible"] = None, False
                elif need is not None and need <= flex:
                    pl["title"] = f"Shock plan: cut ${need:,.0f}/month while the shock lasts to last {target} months"
                    pl["do"] = (f"If the shock hits, lower spending by ${need:,.0f}/month for the "
                                f"{plural(sc['lasts'], 'month') if sc['lasts'] else 'whole time'} it lasts"
                                + (f" ({need / flex:.0%} of spending outside fixed costs)." if flex > 0 else "."))
                    pl["you_c"] = (f"Month by month, {rest}. Spending ${need:,.0f} less each month while the shock "
                                   f"lasts stretches that to {target} months.")
                    pl["apply"] = dict(cut=float(need), label=f"spending cut by ${need:,.0f}/month during the shock")
                    pl["infeasible"] = False
                else:
                    pl["title"] = f"Shock plan: spending cuts alone can't reach {target} months"
                    pl["do"] = pl["you_c"] = (
                        f"Month by month, {rest}. " +
                        (f"Reaching {target} months needs ${need:,.0f}/month while the shock lasts, more than the "
                         f"${flex:,.0f}/month of spending outside fixed costs: the rest must come from extra income "
                         "or outside help." if need is not None else
                         "The cushion goes below $0 as soon as the one-off cost is paid, so a monthly change can't "
                         "fix it: arrange money for the one-off first."))
                    pl["apply"], pl["infeasible"] = None, True
        facts = dict(customer=aid, shock=sc["text"], runway_now=core.runway(p.B, p.I, p.S),
                     runway_after=core.runway(p.B, keep * p.I, p.S + bills * p.H + extra_out, net),
                     survive=target, one_off_only=question == "can_pay", one_off=round(one_off),
                     buffer=round(p.B + windfall),
                     can_pay=None if can_pay is None else can_pay["can_pay"],
                     left=None if can_pay is None else can_pay["left"],
                     month_by_month=None if timed is None else (round(timed["out"], 1) if timed["out"] is not None
                                                                else "does not run out"),
                     steps=[(pl["do"], pl["bank"]) for pl in plans],
                     meets_target=lasts if timed else None,
                     meets=bool(lasts),                 # the one verdict the screen shows; the message uses it too
                     runway_after_raw=None if after.raw is None else float(after.raw),
                     plan=None if confirmed <= 0 else dict(
                         cut=round(cut), extra=round(extra), timed=bool(T),
                         runway=(("no shortfall" if T["plan_out"] is None else f"{T['plan_out']:.1f} months")
                                 if T else runway_json(planned)["display"]),
                         meets=T["plan_lasts"] if T else (planned.status in ("no_shortfall", "capped") or (
                             planned.status == "shortfall" and planned.raw >= target))))
        projection_extra = {"timed": timed["path"] if timed else None}
        return {
            "question": question, "can_pay": can_pay, "timed": timed, "shock_text": sc["text"],
            "risk": {"base": num(risk), "auc": num(core.AUC)},
            "plans": [plan_json(pl) for pl in plans], "timed_basis": bool(T),
            "message": agent.template_message(facts),
            "account_id": aid, "target": target, "plan_active": confirmed > 0,
            "base": runway_json(base), "after": runway_json(after), "planned": runway_json(planned),
            "preview": None if not preview else runway_json(preview), "preview_cut": num(preview_cut),
            "lasts": bool(lasts), "short_by": num(short_by),
            "need": num(need), "confirmed": num(confirmed), "applied_cut": num(cut),
            "entered_cut": num(entered_cut), "cut_clamped": entered_cut > cut, "extra": num(extra),
            "remaining": num(remaining), "one_off": num(one_off),
            "projection": {"months": list(range(horizon + 1)),
                           "today": [num(v) for v in E.projection(base.buffer_left, base.gap, horizon)],
                           "shock": [num(v) for v in E.projection(after.buffer_left, after.gap, horizon)],
                           "plan": ([num(v) for v in T["plan_path"][:horizon + 1]] if T else
                                    [num(v) for v in E.projection(planned.buffer_left, planned.gap, horizon)]),
                           "preview": None if not preview else
                           ([num(v) for v in T["preview_path"][:horizon + 1]] if T else
                            [num(v) for v in E.projection(preview.buffer_left, preview.gap, horizon)]),
                           **projection_extra},
            "fund": fund_json(fund), "fund_after": fund_json(fund_after), "kmon": kmon, "save": num(save),
            "response": {"group": response.group, "label": RP.LABELS[response.group]},
            "guide": {"why": guide.why, "actions": guide.actions, "result": guide.result, "open_gap": guide.open_gap},
            "backup": backup,
        }, facts


def read_fields(text, t, shock, target, p, warn):
    """What the card was read as, field by field, for the person to check before anything is calculated.
    status: "read" (stated in the card), "assumed" (filled in or worked out: the reason says how), "none"."""
    low = text.lower()
    stated = lambda v: v and re.search(rf"(?<![\d.]){v:g}\s*(?:%|percent|per cent)", low)
    F = []

    def add(key, label, unit, status, why=""):
        F.append({"key": key, "label": label, "unit": unit, "value": shock.get(key, target if key == "target" else 0),
                  "status": status, "why": why})

    # money in
    why, st = [], "read"
    if t["job_loss"]:
        ei = float(t["ei_pct"] or 0)
        if ei and not stated(ei):
            st, why = "assumed", why + [f"EI share not given in the card: competition assumption {ei:g}%"]
        elif not ei:
            why.append("job loss with no EI: no money in from that job")
    partner = [u for u in warn if u.startswith("partner")]
    if partner:
        st, why = "assumed", why + partner
    if t.get("extra_income_dollars"):
        st = "assumed"
        why.append(f"${float(t['extra_income_dollars']):+,.0f}/month ÷ usual money in ${float(p.I):,.0f} = "
                   f"{float(t['extra_income_dollars']) / max(float(p.I), 1) * 100:+.1f}% of usual")
    changed_in = t["job_loss"] or t["other_income_change_pct"] or t.get("extra_income_dollars")
    add("keep", "Money in that continues", "%", st if changed_in else "none", "; ".join(why))
    add("bills", "Household bills change", "%", "read" if shock["bills"] else "none")
    why, st = [], "read"
    if t["spending_change_pct"]:
        st = "assumed"
        why.append(f"spending {float(t['spending_change_pct']):+g}% × usual spending ${float(p.S):,.0f} = "
                   f"${float(t['spending_change_pct']) / 100 * float(p.S):+,.0f}/month")
    add("extraOut", "New monthly costs (− = lower)", "$", st if shock["extraOut"] else "none", "; ".join(why))
    why, st = [], "read"
    if t["one_off_months_of_spending"]:
        st = "assumed"
        why.append(f"{float(t['one_off_months_of_spending']):g} month(s) of usual spending × ${float(p.S):,.0f}")
    add("oneOff", "One-time cost", "$", st if shock["oneOff"] else "none", "; ".join(why))
    add("windfall", "One-time money in", "$", "read" if shock["windfall"] else "none")
    add("startIn", "Starts in (months)", "mo", "read" if shock["startIn"] else "none")
    add("lasts", "Lasts (months, 0 = doesn't end)", "mo", "read" if shock["lasts"] else "none")
    after_st, after_why = "none", ""
    if shock["lasts"]:
        back = re.search(r"new job|back to work|re-?employed|find work|finds work|hired again|back to normal|"
                         r"returns? to|goes back", low)
        if t["job_loss"] and not back:
            after_st, after_why = "assumed", "the card doesn't say what happens after: assumed no new job (0%)"
        elif stated(shock["afterPct"]) or back:
            after_st = "read"
        else:
            after_st, after_why = "assumed", f"not stated: assumed {shock['afterPct']:g}% of usual"
    add("afterPct", "Money in after it ends", "%", after_st, after_why)
    add("target", "Target (months)", "mo", "read" if target else "none")
    return F


def plural(n, word):
    return f"{n:g} {word}{'' if n == 1 else 's'}"


def timed_engine(p, after, net, sc, target, confirmed, preview_change):
    """Month-by-month answers for a shock with timing. A monthly change (confirmed cut + extra money in) applies
    while the shock lasts. 'Lasts' = the balance never goes below $0 (including the moment a one-off is paid)
    before the target month."""
    def run(change):
        return core.simulate(p.B, p.I, p.S, after.income + change, after.spending, net, start_in=sc["start_in"],
                             lasts=sc["lasts"], after_frac=sc["after"])

    def ok(o):
        return o is None or o >= target

    out, path = run(0.0)
    lasts = ok(out)
    # lowest point within the target window, including right after a one-off is paid
    lows = list(path[1:target + 1])
    if sc["start_in"] < target:
        lows.append(path[sc["start_in"]] - net)
    low = min(lows) if lows else 0.0
    short_by = 0.0 if lasts else max(-float(low), 0.0)
    if lasts:
        need = 0.0
    else:
        hi = 10.0 * max(after.spending, p.S, 1.0)
        if not ok(run(hi)[0]):
            need = None                      # a monthly change can't fix it (e.g. the one-off alone is too big)
        else:
            lo = 0.0
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(run(mid)[0]) else (mid, hi)
            need = math.ceil(hi * 100) / 100
    plan_out, plan_path = run(confirmed) if confirmed > 0 else (out, path)
    pv_out, pv_path = run(preview_change) if preview_change > 0 else (None, None)
    buffer_gone = p.B - (net if sc["start_in"] == 0 else 0) <= 0
    group = "A" if buffer_gone else ("C" if lasts else "D")
    return dict(out=out, path=path, lasts=bool(lasts), short_by=short_by, need=need, low=low,
                plan_out=plan_out, plan_path=plan_path, plan_lasts=bool(ok(plan_out)),
                preview_out=pv_out, preview_path=pv_path, group=group)


def timed_texts(T, sc, target, confirmed, need, remaining, loan):
    """Guide and backup steps for a timed shock, every number from the same month-by-month engine."""
    when = "starts now" if not sc["start_in"] else f"starts in {sc['start_in']} months"
    ends = (f"lasts {plural(sc['lasts'], 'month')}, then money in returns to {sc['after'] * 100:g}% of usual" if sc["lasts"]
            else "doesn't end")
    lasts_txt = ("the cushion does not run out" if T["out"] is None
                 else f"the cushion runs out in about {T['out']:.1f} months")
    why = [f"Counting month by month (the shock {when} and {ends}), {lasts_txt}.",
           (f"That covers the {target}-month target." if T["lasts"] else
            f"That falls short of the {target}-month target: at its lowest before month {target}, the balance is "
            f"{money(T['low'])}.")]
    actions, result = [], []
    if T["lasts"]:
        actions.append("No monthly change is needed to reach the target with this timing. Keep the cushion for "
                       "surprises.")
    elif need is None:
        actions.append("A monthly change during the shock can't reach the target (the cushion goes below $0 as soon "
                       "as the one-off cost is paid). Arrange money for the one-off first.")
    else:
        actions.append(f"Find {money(need)}/month while the shock lasts (cuts the customer confirms, or extra money "
                       f"in) to reach {target} months.")
    if confirmed > 0:
        plan_txt = ("the cushion does not run out" if T["plan_out"] is None
                    else f"the cushion runs out in about {T['plan_out']:.1f} months")
        result.append(f"With the {money(confirmed)}/month you confirmed, {plan_txt} (month by month), "
                      + ("meeting the target." if T["plan_lasts"] else "still short of the target.")
                      + (f" Still needed: {money(remaining)}/month." if remaining and remaining > 0.5 else ""))
    elif not T["lasts"]:
        result.append("Nothing confirmed yet, so the shortfall is still open.")
    guide = PG.Guide(T["group"], RP.LABELS[T["group"]], why=why, actions=actions, result=result,
                     open_gap=bool(not T["lasts"] and (remaining is None or remaining > 0.5)))
    backup = [why[0] + " " + why[1]]
    if not T["lasts"]:
        if confirmed <= 0:
            backup.append("You haven't confirmed any cuts or extra money in yet, so the shortfall is still open.")
        else:
            backup.append(result[0])
        if need is not None and remaining and remaining > 0.5:
            backup.append(f"To last {target} months, you still need another {money(remaining)}/month while the "
                          "shock lasts (more cuts, more money in, or outside help).")
    if loan > 0:
        backup.append(f"The loan payment of {money(loan)}/month is part of money out. Ask the lender early about "
                      "payment options.")
    return guide, backup


IF_SHOCK = ("Shock plan", "Refill the", "Protect the loan payment")


def plan_when(title: str) -> str:
    """Two kinds of plan, never mixed: what to do NOW on today's income (build the cushion, set things up), and
    what to do IF the shock hits (on the post-shock cash flow)."""
    return "if_shock" if title.startswith(IF_SHOCK) else "now"


def plan_json(pl: dict) -> dict:
    text = pl["pill"][0]
    kind = ("safety" if pl.get("pinned") else "support" if pl.get("infeasible")
            else "rank" if pl.get("sig") else "low")
    return {"pill": text, "kind": kind, "title": pl["title"], "do": pl["do"], "you": pl["you"], "you_c": pl["you_c"],
            "data": pl["data"], "bank": pl["bank"], "start": num(pl.get("start")),
            "new": num(pl["new"]) if pl["new"] is not None else None,
            "red": num(pl["red"]) if pl["red"] is not None else None, "test": pl.get("test", ""), "track": pl.get("track", ""),
            "apply": None if not pl.get("apply") else {k: (num(v) if k != "label" else v) for k, v in pl["apply"].items()},
            "goal": num(pl.get("goal")), "goal_note": goal_note(pl.get("goal")), "when": plan_when(pl["title"]), 
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
        d.append(f"lasts {plural(sc['lasts'], 'month')}")
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
