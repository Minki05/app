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

srv.shutdown()
print(f"\nResult: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
