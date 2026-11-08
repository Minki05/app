"""Checks for the web app's real request path: site/web_app.py Store + HTTP handler, agent.py, core.py.

Run:  .venv/bin/python site/check_web.py
- Starts the real handler on a free 127.0.0.1 port inside this process and stops it at the end.
- Gemini is never called: agent._gemini is replaced by fakes (and by a tripwire everywhere else).
- Writes no files.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["GEMINI_API_KEY"] = ""          # stops web_app from loading the key from secrets.toml

sys.path.append(str(HERE.parent))          # agent.py / core.py live in the repo root (as in web_app.py)
import agent  # noqa: E402


def _tripwire(*a, **k):
    raise AssertionError("Gemini was called")


agent._gemini = _tripwire

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import engine as E  # noqa: E402
import web_app as W  # noqa: E402

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -> {detail}"))


STORE = W.load_store()
srv = W.QuietServer(("127.0.0.1", 0), W.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"


def post(path, body, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


B20111 = float(STORE.prof.loc[20111].B)

# ---------------------------------------------------------------------------------------------------------
print("\n1) Cost exactly equal to cash: can pay, nothing left (runway stays 0)")
code, r = post("/api/simulate", {"account_id": 20111, "one_off": 17611.863333333335, "question": "can_pay"})
cp = r.get("can_pay") or {}
check("request answered 200", code == 200, code)
check("have == cost, left == 0", cp.get("have") == cp.get("cost") and cp.get("left") == 0, cp)
check("can_pay true, cash_left false", cp.get("can_pay") is True and cp.get("cash_left") is False, cp)
check("official runway after stays 0 (buffer_gone)", r["after"]["status"] == "buffer_gone" and r["after"]["raw"] == 0,
      r["after"])
check("template says it can be paid with no cushion left",
      "can pay" in r["message"] and "no cash cushion" in r["message"] and "would not cover" not in r["message"],
      r["message"])
check("plan text does not call an equal cost 'larger'", all("larger than" not in p["you"] for p in r["plans"]),
      [p["you"] for p in r["plans"]])

for delta, want_can, want_text in ((0.3, False, "less than $1 short"), (-0.3, True, "less than $1 left"),
                                   (-500.0, True, "$500 left"), (500.0, False, "$500 short")):
    code, r = post("/api/simulate", {"account_id": 20111, "one_off": B20111 + delta, "question": "can_pay"})
    cp = r["can_pay"]
    check(f"cost = cash {delta:+g}: can_pay {want_can}, message '{want_text}'",
          cp["can_pay"] is want_can and want_text in r["message"], (cp, r["message"]))

captured = {}


def fake_gemini(prompt, want_json=False):
    captured["prompt"] = prompt
    return "FAKE MESSAGE"


agent._gemini = fake_gemini
code, m = post("/api/message", {"account_id": 20111, "one_off": 17611.863333333335, "question": "can_pay"})
facts = json.loads(captured["prompt"].split("Facts (JSON): ", 1)[1])
check("AI facts: paid from savings, no cushion left", "can be paid" in facts["outlook_after_shock"]
      and "no cash cushion" in facts["outlook_after_shock"], facts["outlook_after_shock"])
check("AI message endpoint returns the fake answer", code == 200 and m["message"] == "FAKE MESSAGE", (code, m))
agent._gemini = _tripwire

# furnace preset (cost = one month of spending) still shows money left, as before
S20111 = float(STORE.prof.loc[20111].S)
code, r = post("/api/simulate", {"account_id": 20111, "keep": 100, "one_off": S20111, "question": "can_pay"})
check("furnace preset unchanged: can pay with cash left, runway 4.0",
      r["can_pay"]["cash_left"] is True and r["after"]["display"] == "4.0", (r["can_pay"], r["after"]["display"]))

# ---------------------------------------------------------------------------------------------------------
print("\n2) Input validation: structured 400s, 404 kept, normal page requests still 200")
bad = [
    ("JSON array body", b"[]", None),
    ("JSON string body", b'"x"', None),
    ("not JSON", b"{oops", None),
    ("target = Infinity (1e999)", b'{"account_id":20111,"target":1e999}', "target"),
    ("target = NaN literal", b'{"account_id":20111,"target":NaN}', "target"),
    ("keep = -Infinity", b'{"account_id":20111,"keep":-Infinity}', "keep"),
]
for name, body, field in bad:
    code, r = post("/api/simulate", body, raw=True)
    check(f"{name} -> 400 field={field}", code == 400 and r.get("field") == field and r.get("error"), (code, r))
bad_obj = [
    ("checked = 'false' (string)", {"account_id": 20111, "checked": "false", "cut": 1000}, "checked"),
    ("checked = 1", {"account_id": 20111, "checked": 1, "cut": 1000}, "checked"),
    ("keep = true (bool as number)", {"account_id": 20111, "keep": True}, "keep"),
    ("cut = '1000' (string)", {"account_id": 20111, "checked": True, "cut": "1000"}, "cut"),
    ("target = 6.5 (not whole)", {"account_id": 20111, "target": 6.5}, "target"),
    ("target = 0 (below range)", {"account_id": 20111, "target": 0}, "target"),
    ("keep = 151 (above range)", {"account_id": 20111, "keep": 151}, "keep"),
    ("one_off = -5 (below range)", {"account_id": 20111, "one_off": -5}, "one_off"),
    ("account_id = true", {"account_id": True}, "account_id"),
    ("account_id = 20111.5", {"account_id": 20111.5}, "account_id"),
    ("account_id missing", {}, "account_id"),
    ("question = 5", {"account_id": 20111, "question": 5}, "question"),
]
for name, body, field in bad_obj:
    code, r = post("/api/simulate", body)
    check(f"{name} -> 400 field={field}", code == 400 and r.get("field") == field, (code, r))
# huge integers: exact int range check, never a float conversion (was OverflowError)
for k, v in (("target", 10 ** 400), ("target", -10 ** 400), ("cut", 10 ** 400), ("keep", -10 ** 400)):
    try:
        W.validate_request("/api/simulate", {"account_id": 20111, k: v})
        res = "accepted"
    except W.BadRequest as e:
        res = e.field
    except Exception as e:  # noqa: BLE001 - the regression being checked
        res = f"{type(e).__name__}"
    check(f"validate_request {k}={'-' if v < 0 else ''}10**400 -> range error on {k}", res == k, res)
    code, r = post("/api/simulate", f'{{"account_id":20111,"{k}":{v}}}'.encode(), raw=True)
    check(f"HTTP {k}={'-' if v < 0 else ''}10**400 -> 400 field={k}", code == 400 and r.get("field") == k
          and "between" in r.get("error", ""), (code, r))
code, r = post("/api/simulate", b'{"account_id":20111,"target":' + b"9" * 5000 + b"}", raw=True)
check("HTTP 5,000-digit integer -> structured 400", code == 400 and r.get("error"), (code, r))
code, r = post("/api/simulate", f'{{"account_id":{10 ** 400}}}'.encode(), raw=True)
check("HTTP account_id=10**400 -> 404 (no such account)", code == 404, (code, r))
code, r = post("/api/simulate", {"account_id": 20111, "target": 12, "keep": 0, "cut": 10 ** 9, "checked": True})
check("largest allowed integers still 200", code == 200, (code, r))
code, r = post("/api/twist", {"account_id": 20111, "text": 123})
check("twist text = 123 -> 400 field=text", code == 400 and r.get("field") == "text", (code, r))
code, r = post("/api/message", [])
check("message with [] body -> 400", code == 400, (code, r))
code, r = post("/api/simulate", {"account_id": 99999999})
check("unknown account -> 404 (unchanged)", code == 404, (code, r))
code, r = post("/api/simulate", {"account_id": "20111"})
check("account_id as digit string still accepted", code == 200, (code, r))
code, r = post("/api/simulate", {"account_id": 20111, "checked": False, "cut": 1000})
check("checked = false really ignores the cut", code == 200 and r["confirmed"] == 0 and r["applied_cut"] == 0,
      (code, r.get("confirmed")))
code, r = post("/api/simulate", {"account_id": 20111, "checked": True, "cut": 1000})
check("checked = true applies the $1,000 cut", code == 200 and r["confirmed"] == 1000, (code, r.get("confirmed")))
page_body = {"account_id": 20111, "keep": 55, "bills_pct": 15, "one_off": 0, "extra_out": 0, "windfall": 0,
             "start_in": 0, "lasts": 0, "after_pct": 100, "question": "last_months", "label": "job loss (EI 55%)",
             "target": 6, "checked": False, "cut": 0, "extra": 0, "kmon": 3, "save": 0, "plan_buffer": 0}
code, r = post("/api/simulate", page_body)
check("page's own request body (EI 55% + bills 15%) -> 200, runway 3.4",
      code == 200 and r["after"]["display"] == "3.4", (code, r.get("after")))
code, r = post("/api/simulate", {**page_body, "extra_out": -1e9, "target": None, "label": None})
check("page extremes (typed -1e9 monthly cost, null target/label) -> 200", code == 200, (code, r))

# ---------------------------------------------------------------------------------------------------------
print("\n3) Negation: 'I did not lose my job' is not a job loss; Gemini's correct answer is not overwritten")
REPRO = "I did not lose my job. My rent goes up 10%."
r = agent.parse_rules(REPRO)
check("rules: job_loss false, EI not set, rent +10% kept",
      r["job_loss"] is False and r["ei_pct"] is None and r["bills_up_pct"] == 10.0, r)
for text in ("I didn't lose my job, but my rent goes up 10%.", "I wasn't laid off. Bills go up 10%.",
             "We have not lost our jobs; our rent goes up 10%."):
    r = agent.parse_rules(text)
    check(f"rules: {text[:45]!r} -> no job loss, bills +10%", not r["job_loss"] and r["bills_up_pct"] == 10.0, r)
r = agent.parse_rules("My husband did not lose his job. Our rent goes up 10%.")
check("rules: partner did not lose job -> no income change, no assumed share",
      r["other_income_change_pct"] == 0 and not r["unsupported"] and r["bills_up_pct"] == 10.0, r)
for text, want in (("I lost my job. EI pays 55%. My rent goes up 10%.", dict(job_loss=True, ei_pct=55.0, bills_up_pct=10.0)),
                   ("I lost my job, no EI.", dict(job_loss=True, ei_pct=0.0)),
                   ("My husband lost his job; he earns 40% of our income.", dict(job_loss=False, other_income_change_pct=-40.0)),
                   ("They lose their job (EI at 55%) and their household bills go up 15% in the same month.",
                    dict(job_loss=True, ei_pct=55.0, bills_up_pct=15.0))):
    r = agent.parse_rules(text)
    check(f"rules unchanged: {text[:45]!r}", {k: r[k] for k in want} == want, r)

# an EI share must never turn a denied job loss back into one, and denied / what-if EI is not received
EI_CASES = [
    ("I have not lost my job and do not receive EI at 55%. My rent goes up 10%.",
     dict(job_loss=False, ei_pct=None, bills_up_pct=10.0, other_income_change_pct=0.0), False),
    ("I did not lose my job. My rent goes up 10%.",
     dict(job_loss=False, ei_pct=None, bills_up_pct=10.0, other_income_change_pct=0.0), False),
    ("I did not lose my job. If I did, EI would pay 55% of my income. My rent goes up 10%.",
     dict(job_loss=False, ei_pct=None, bills_up_pct=10.0, other_income_change_pct=0.0), False),
    ("I did not lose my job. My hours are cut by 20%.",
     dict(job_loss=False, ei_pct=None, bills_up_pct=0.0, other_income_change_pct=-20.0), False),
    ("I lost my job. EI pays 55%. My rent goes up 10%.", dict(job_loss=True, ei_pct=55.0, bills_up_pct=10.0), False),
    ("I lost my job, no EI.", dict(job_loss=True, ei_pct=0.0), False),
    ("I lost my job and do not receive EI at 55%.", dict(job_loss=True, ei_pct=0.0), False),
    ("If they lose their job, EI pays 55% for 6 months. Do they make it?", dict(job_loss=True, ei_pct=55.0), False),
    ("My husband lost his job; he earns 40% of our income.",
     dict(job_loss=False, ei_pct=None, other_income_change_pct=-40.0), False),
    ("I lost my job and get about 55% of my pay from EI. My rent and utilities are going up 15%.",
     dict(job_loss=True, ei_pct=55.0, bills_up_pct=15.0), False),
    ("I did not lose my job, but EI pays 55% of my income.", dict(job_loss=False, ei_pct=None), True),
    ("I was not fired. I was laid off and my rent goes up 10%.",
     dict(job_loss=False, ei_pct=None, bills_up_pct=10.0), True),
]
for text, want, warns in EI_CASES:
    r = agent.parse_rules(text)
    check(f"rules: {text[:58]!r}", {k: r[k] for k in want} == want and bool(r["unsupported"]) == warns,
          ({k: r[k] for k in want}, r["unsupported"]))

G = dict(job_loss=False, ei_pct=None, other_income_change_pct=0, bills_up_pct=10, extra_monthly_dollars=0,
         spending_change_pct=0, one_off_dollars=0, one_off_months_of_spending=0, windfall_dollars=0, starts_in_months=0,
         shock_lasts_months=0, income_after_pct=100, survive_months=None, question="auto", unsupported=[])


def gemini_returns(d):
    agent._gemini = lambda prompt, want_json=False: json.dumps(d)


gemini_returns(G)
d, src = agent.parse_twist(REPRO)
check("Gemini says no job loss -> kept (no rule overwrite), nothing added",
      src == "Gemini" and d["job_loss"] is False and not agent.ADDED and d["bills_up_pct"] == 10, (d, agent.ADDED))
code, t = post("/api/twist", {"account_id": 20111, "text": REPRO})
check("/api/twist: money in stays 100%, bills +10%, no warning",
      code == 200 and t["shock"]["keep"] == 100 and t["shock"]["bills"] == 10 and t["warn"] == [], (code, t))
gemini_returns({**G, "job_loss": True, "ei_pct": 55})
d, _ = agent.parse_twist(REPRO)
check("Gemini says job loss but card denies it -> not applied, warned",
      d["job_loss"] is False and any("no job loss" in w for w in d["unsupported"]), d)
gemini_returns({**G, "bills_up_pct": 0})
d, _ = agent.parse_twist("I was not fired. I was laid off and my rent goes up 10%.")
check("card both denies and states a job loss -> not applied as fact, warned",
      d["job_loss"] is False and any("unclear" in w for w in d["unsupported"]), d)
d, _ = agent.parse_twist("I was not fired. I was laid off and my rent goes up 10%.")
check("mixed card: Gemini said no job loss -> stays off, warned once",
      d["job_loss"] is False and sum("unclear" in w for w in d["unsupported"]) == 1, d)
gemini_returns({**G, "job_loss": True, "ei_pct": 55})
d, _ = agent.parse_twist("I was not fired. I was laid off and my rent goes up 10%.")
check("mixed card: Gemini said job loss -> not applied as fact, warned",
      d["job_loss"] is False and any("unclear" in w for w in d["unsupported"]), d)
# Gemini reads the EI-negation cards correctly (no job loss): the rule check must not undo that
EI_TEXTS = [c[0] for c in EI_CASES[:4]]
for text in EI_TEXTS:
    want_bills = 10 if "rent" in text else 0
    want_inc = -20 if "hours" in text else 0
    gemini_returns({**G, "bills_up_pct": want_bills, "other_income_change_pct": want_inc})
    d, _ = agent.parse_twist(text)
    check(f"Gemini 'no job loss' kept: {text[:45]!r}",
          d["job_loss"] is False and d["ei_pct"] is None and not agent.ADDED and not d["unsupported"]
          and d["bills_up_pct"] == want_bills and d["other_income_change_pct"] == want_inc, (d, agent.ADDED))
    gemini_returns({**G, "bills_up_pct": 0, "other_income_change_pct": 0})    # Gemini missed the other shocks
    d, _ = agent.parse_twist(text)
    check(f"rules fill only the real shock, not a job loss: {text[:30]!r}",
          d["job_loss"] is False and d["bills_up_pct"] == want_bills and d["other_income_change_pct"] == want_inc, d)
gemini_returns({**G, "job_loss": True, "ei_pct": 55})
d, _ = agent.parse_twist(EI_TEXTS[0])
check("Gemini wrongly says job loss + EI 55% on the EI-negation card -> not applied, warned",
      d["job_loss"] is False and d["ei_pct"] is None and d["unsupported"], d)
gemini_returns(G)
code, t = post("/api/twist", {"account_id": 20111, "text": EI_TEXTS[0]})
check("/api/twist EI-negation card: money in 100%, bills +10%, no warning",
      code == 200 and t["shock"]["keep"] == 100 and t["shock"]["bills"] == 10 and t["warn"] == [], (code, t))
agent._gemini = lambda prompt, want_json=False: None                 # Gemini unavailable -> rule reader
code, t = post("/api/twist", {"account_id": 20111, "text": EI_TEXTS[0]})
check("/api/twist EI-negation card with rule reader only: money in 100%, bills +10%",
      code == 200 and t["source"] == "rule-based reader" and t["shock"]["keep"] == 100
      and t["shock"]["bills"] == 10, (code, t))
gemini_returns({**G, "bills_up_pct": 0})
d, _ = agent.parse_twist("They lose their job (EI at 55%) and their household bills go up 15% in the same month.")
check("Gemini missed a clearly stated job loss -> rule check still fills it (listed)",
      d["job_loss"] is True and d["ei_pct"] == 55 and agent.ADDED, (d, agent.ADDED))
agent._gemini = _tripwire

# ---------------------------------------------------------------------------------------------------------
print("\n4) Bank support is phrased as something to ask about, never approved")
code, r = post("/api/simulate", {"account_id": 20111, "one_off": 30000, "question": "can_pay"})
msg = r["message"]
check("template has no 'We can pre-approve' / 'We can'", "We can" not in msg, msg)
check("template asks instead", "Ask us whether we can pre-approve" in msg, msg)
steps = [p["do"] for p in r["plans"][:3]]
check("step text (numbers, conditions) kept word for word", all(s in msg for s in steps), steps)
check("shortfall amount kept", f"${30000 - B20111:,.0f} short" in msg, msg)
agent._gemini = fake_gemini
post("/api/message", {"account_id": 20111, "one_off": 30000, "question": "can_pay"})
facts = json.loads(captured["prompt"].split("Facts (JSON): ", 1)[1])
check("AI facts: no 'We can', bank steps start with 'Ask us whether'",
      all("We can" not in s and "Ask us whether we can" in s for s in facts["steps_in_order"]), facts["steps_in_order"])
check("AI prompt forbids approval wording", "never say the bank has approved" in captured["prompt"])
agent._gemini = _tripwire

# ---------------------------------------------------------------------------------------------------------
print("\n5) Message facts belong to the request (no state shared between requests)")
check("Store keeps no per-account facts", not hasattr(STORE, "_facts"))
QA = {"account_id": 20111, "keep": 55, "label": "scenario A"}
QB = {"account_id": 20111, "one_off": 5000, "label": "scenario B"}
real_template, real_write = agent.template_message, agent.write_message
state = {"nested": False}


def interleaving_template(facts):
    # another request for the same account runs while A is still being built
    if facts["shock"] == "scenario A" and not state["nested"]:
        state["nested"] = True
        STORE.simulate(QB)
    return real_template(facts)


agent.template_message = interleaving_template
agent.write_message = lambda facts: (facts["shock"], "template")
check("A's message uses A's facts even if B ran in between", STORE.message(QA)["message"] == "scenario A")
agent.template_message = real_template

results = {}


def worker(i):
    q, want = (QA, "scenario A") if i % 2 else (QB, "scenario B, one-off cost $5,000")
    code, m = post("/api/message", q)
    results[i] = (want, code, m.get("message"))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
[t.start() for t in threads]
[t.join() for t in threads]
check("40 concurrent /api/message calls, 2 scenarios, same account: each gets its own facts",
      len(results) == 40 and all(c == 200 and lab == msg for lab, c, msg in results.values()),
      [v for v in results.values() if v[0] != v[2]][:3])
agent.write_message = real_write

# ---------------------------------------------------------------------------------------------------------
print("\n6) site/engine.py against an independent copy of the README formula (all accounts)")
ms = pd.read_csv(E.DATA_DIR / "monthly_summary.csv").sort_values(["account_id", "month"])
g = ms.groupby("account_id")
ref = pd.DataFrame({"I": g.money_in.median(), "S": g.money_out.median(), "H": g.household_bills.median(),
                    "B": g.avg_daily_balance.apply(lambda s: s.iloc[-6:].mean())})


def readme_runway(B, inc, sp, one_off=0.0):
    left, gap = B - one_off, sp - inc
    if left <= 0:
        return "0.0"
    if gap <= 0:
        return "no shortfall (120+)"
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(min(left / gap, 120))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


prof = STORE.prof
check("profiles: I, S, H, B match the README definitions (4,500 accounts)",
      len(ref) == len(prof) == 4500 and all(np.allclose(ref[k], prof.loc[ref.index, k], rtol=0, atol=1e-9)
                                            for k in "ISHB"))
scen = {"normal": lambda a: (a.I, a.S, 0.0, E.Shock()),
        "EI 55%": lambda a: (0.55 * a.I, a.S, 0.0, E.Shock(0.55)),
        "one-off = S": lambda a: (a.I, a.S, a.S, E.Shock(one_off=a.S)),
        "EI 55% + bills 15%": lambda a: (0.55 * a.I, a.S + 0.15 * a.H, 0.0, E.Shock(0.55, 0.15))}
for name, f in scen.items():
    bad_ids = []
    for aid, a in ref.iterrows():
        inc, sp, c, shock = f(a)
        p = prof.loc[aid]
        got = E.runway(p.B, p.I, p.S, p.H, shock)
        want = readme_runway(a.B, inc, sp, c)
        if (got.display if got.status != "capped" else "120.0") != want:
            bad_ids.append((aid, got.display, want))
    check(f"{name}: engine display == README formula for all 4,500", not bad_ids, bad_ids[:5])

# ---------------------------------------------------------------------------------------------------------
print("\n7) Future savings goals are never added to today's cash")
EI55 = {"account_id": 20111, "keep": 55}
code, base55 = post("/api/simulate", EI55)
A = base55["after"]
check("20111 keep 55%: starting cash $17,611.86, runway 3.4",
      code == 200 and abs(A["buffer_left"] - 17611.86) < 0.01 and A["display"] == "3.4", A)
check("…and the plan line is the same cash, not active",
      base55["planned"]["buffer_left"] == A["buffer_left"] and base55["planned"]["display"] == "3.4"
      and base55["plan_active"] is False and "plan_buffer" not in base55, (base55["planned"], base55["plan_active"]))
fund = [p for p in base55["plans"] if p["title"].startswith("Build an emergency fund")]
check("emergency fund plan kept ($8,187 to save), no chart apply, goal note sent",
      len(fund) == 1 and abs(fund[0]["goal"] - 8186.95) < 0.01 and fund[0]["apply"] is None
      and "isn't saved yet" in fund[0]["goal_note"] and "$8,187" in fund[0]["goal_note"], fund)
_, rep = post("/api/simulate", {"account_id": 20111, "keep": 100, "one_off": 5000, "question": "can_pay"})
refill = [p for p in rep["plans"] if p["title"].startswith("Refill")]
check("repair refill plan kept ($5,000 goal), no chart apply, goal note",
      len(refill) == 1 and refill[0]["goal"] == 5000 and refill[0]["apply"] is None and refill[0]["goal_note"], refill)
LOAN = 24640
_, rl = post("/api/simulate", {"account_id": LOAN, "keep": 55})
loanp = [p for p in rl["plans"] if p["title"] == "Protect the loan payment"]
check(f"loan protection plan ({LOAN}) kept, no chart apply, goal note",
      len(loanp) == 1 and loanp[0]["goal"] > 0 and loanp[0]["apply"] is None and loanp[0]["goal_note"], loanp)
# every plan on a broad sample: any 'apply' left is a monthly cut only
bad_apply = []
for aid in list(STORE.prof.index[::90]) + [20111, LOAN]:
    for q in ({"keep": 55}, {"keep": 100, "one_off": 3000}, {"keep": 55, "bills_pct": 15}):
        _, rr = post("/api/simulate", {"account_id": int(aid), **q})
        bad_apply += [(aid, p["title"], p["apply"]) for p in rr["plans"] if p["apply"] and set(p["apply"]) != {"cut", "label"}]
check("no plan on 52 accounts × 3 shocks offers to add savings to cash (apply = cut only)", not bad_apply, bad_apply[:3])
for path in ("/api/simulate", "/api/message"):
    code, r = post(path, {**EI55, "plan_buffer": 8186.951666666664})
    check(f"{path} plan_buffer 8186.95 -> 400 with explanation",
          code == 400 and r.get("field") == "plan_buffer" and "still to be saved" in r.get("error", ""), (code, r))
for v, want in ((0, 200), (None, 200), (-1, 400), (10 ** 400, 400)):
    code, r = post("/api/simulate", {**EI55, "plan_buffer": v})
    ok = code == want and (want != 200 or r["after"] == A and r["planned"] == base55["planned"])
    check(f"plan_buffer={'10**400' if v == 10 ** 400 else v!r} -> {want}", ok, (code, r.get("error")))
code, r = post("/api/simulate", b'{"account_id":20111,"keep":55,"plan_buffer":NaN}', raw=True)
check("plan_buffer NaN -> 400 field=plan_buffer", code == 400 and r.get("field") == "plan_buffer", (code, r))
code, r = post("/api/simulate", {**EI55, "save": 5000, "kmon": 6})
check("monthly saving $5,000 alone: same cash and runway, plan not active, only the fund timeline moves",
      code == 200 and r["base"] == base55["base"] and r["after"] == A and r["planned"] == base55["planned"]
      and r["plan_active"] is False and r["fund"]["months"] is not None, (r["after"], r["plan_active"], r["fund"]))
code, r = post("/api/simulate", {**EI55, "save": 5000})    # same 3-month goal: only the saving amount differs
check("customer message and plans unchanged by a monthly saving amount alone",
      r["message"] == base55["message"] and r["plans"] == base55["plans"] and r["after"] == A)
code, r = post("/api/simulate", {**EI55, "checked": True, "cut": 1000})
P_ = r["planned"]
check("confirmed $1,000/month cut: same starting cash, monthly gap $1,000 smaller, plan active",
      code == 200 and P_["buffer_left"] == A["buffer_left"] and abs((A["gap"] - P_["gap"]) - 1000) < 1e-6
      and P_["raw"] > A["raw"] and r["plan_active"] is True, (P_, A))
code, r = post("/api/simulate", {**EI55, "extra": 500})
check("extra money in $500: same starting cash, gap $500 smaller",
      r["planned"]["buffer_left"] == A["buffer_left"] and abs((A["gap"] - r["planned"]["gap"]) - 500) < 1e-6, r["planned"])
code, r = post("/api/simulate", {**EI55, "windfall": 3000})
check("one-off money in (windfall) $3,000 still adds to cash, as before",
      abs(r["after"]["buffer_left"] - (B20111 + 3000)) < 1e-6 and r["after"]["display"]
      == readme_runway(B20111 + 3000, A["income"], A["spending"]), r["after"])
NEG = int(STORE.prof[STORE.prof.B <= 0].index[0])
code, r = post("/api/simulate", {"account_id": NEG, "keep": 55, "save": 1000})
check(f"no-cushion account {NEG}: still 0 months, plan line not lifted by any savings goal",
      code == 200 and r["after"]["status"] == "buffer_gone" and r["planned"]["status"] == "buffer_gone"
      and r["planned"]["buffer_left"] == r["after"]["buffer_left"] <= 0 and r["plan_active"] is False
      and all(p["apply"] is None or set(p["apply"]) == {"cut", "label"} for p in r["plans"]), r["planned"])

# the page's own plan-card code (site/web/app.js), run without a browser on the API's real plans
import shutil  # noqa: E402
import subprocess  # noqa: E402

node = shutil.which("node")
if not node:
    print("  SKIP  page code check: node not installed")
else:
    src = (HERE / "web" / "app.js").read_text()
    snippet = src[src.index("  const isApplied"):src.index("  // One plan per row")]
    harness = """
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const state = {checked: false, cut: 0, target: 6};
const planFor = () => state; const reduced = true; let sims = 0;
const syncControls = () => {}; const simulate = () => { sims++; }; const goStep = () => {};
const $ = () => ({scrollIntoView() {}});
const data = JSON.parse(require("fs").readFileSync(0, "utf8"));
const S = {acct: 20111, sim: data, preview: new Map()};
""" + snippet + """
const out = {rows: [], sentBefore: sims};
for (const pl of data.plans) {
  const key = pl.title.replace(/[\\d$,.%]+/g, "#");
  const html = applyRow(pl, key);
  applyPlan(key);
  const afterPreview = {...state}, preview = S.preview.get(20111) || 0;
  confirmPlan(key);
  out.rows.push({title: pl.title, button: html.includes("Preview this cut on the chart"), note: html.includes("isn't saved yet"),
                 changesRisk: html.includes("Changes risk"), afterPreview, preview, state: {...state}});
  state.checked = false; state.cut = 0; S.preview.clear();
}
out.state = state; out.sims = sims;
console.log(JSON.stringify(out));
"""
    res = subprocess.run([node, "-e", harness], input=json.dumps(base55), capture_output=True, text=True, timeout=30)
    try:
        js = json.loads(res.stdout)
    except ValueError:
        js = None
    check("page code runs on the real API plans", js is not None, res.stderr[-300:])
    if js:
        rows = {r["title"]: r for r in js["rows"]}
        f_ = rows.get(fund[0]["title"], {})
        check("page: savings goal shows the note, no 'Show this plan' button, no 'Changes risk'",
              f_.get("note") and not f_.get("button") and not f_.get("changesRisk"), f_)
        check("page: clicking the savings goal changes nothing (no cash, no cut)",
              f_.get("state") == {"checked": False, "cut": 0, "target": 6} and not f_.get("preview"), f_)
        cutp = [r for t, r in rows.items() if t.startswith("Shock plan: cut")]
        check("page: 'Preview' only draws the cut: nothing is confirmed or counted",
              cutp and cutp[0]["button"] and cutp[0]["preview"] > 0
              and cutp[0]["afterPreview"] == {"checked": False, "cut": 0, "target": 6}, cutp)
        check("page: 'Customer confirms this cut' is what puts the cut in the plan (and only the cut)",
              cutp and cutp[0]["state"]["checked"] is True and cutp[0]["state"]["cut"] == cutp[0]["preview"]
              and set(cutp[0]["state"]) == {"checked", "cut", "target"}, cutp)
    body = src[src.index("function simBody"):src.index("// ------", src.index("function simBody"))]
    check("page request body no longer carries plan_buffer", "plan_buffer" not in body and "planBuffer" not in src)

srv.shutdown()
print(f"\nResult: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
