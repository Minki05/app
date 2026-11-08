"""Demo-readiness guardrails. Run:  python site/check_guardrails.py

Each section is one failure we must not show judges. Gemini is switched off for the whole run, so every check
also proves the app works end to end without AI (rule-based reader + template message).
"""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["GEMINI_API_KEY"] = ""
sys.path.append(str(HERE.parent))
import agent  # noqa: E402

agent._gemini = lambda *a, **k: None          # no AI anywhere in this run
import web_app  # noqa: E402

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed += bool(ok)
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ->  {detail}"))


S = web_app.load_store()
JS = (HERE / "web" / "app.js").read_text()
ACCOUNTS = [20111, 21786, 55447, 20133, 42958, 65739, 28048, 62195, 89952]


def texts(d):
    out = list(d["backup"]) + d["guide"]["why"] + d["guide"]["actions"] + d["guide"]["result"] + [d["message"]]
    for p in d["plans"]:
        out += [p["title"], p["do"], p["you"], p.get("you_c", ""), p["data"], p["bank"], p.get("test", ""),
                p.get("track", ""), p.get("goal_note", "") or ""]
    return " ".join(x for x in out if x)


def money(x):
    return f"${x:,.0f}"


print("\n1) Every screen gives the same answer when a plan is applied")
for acc in ACCOUNTS:
    d0 = S.simulate(dict(account_id=acc, keep=55, bills_pct=15, target=6))
    cut_plans = [p for p in d0["plans"] if p.get("apply") and p["apply"].get("cut")]
    if not cut_plans:
        continue
    cut = math.ceil(cut_plans[0]["apply"]["cut"])
    d = S.simulate(dict(account_id=acc, keep=55, bills_pct=15, target=6, checked=True, cut=cut))
    pl = d["planned"]
    # chart: the plan line must cross $0 in the month the headline says
    path = d["projection"]["plan"]
    if pl["status"] == "shortfall" and pl["raw"] and pl["raw"] < len(path) - 1:
        first_neg = next((i for i, v in enumerate(path) if v is not None and v < 0), None)
        check(f"{acc}: chart line runs out in the month of the plan's runway ({pl['raw']:.1f})",
              first_neg == math.ceil(pl["raw"]) or first_neg == math.floor(pl["raw"]) + 1, f"chart month {first_neg}")
    # text: what's still needed must match need - confirmed
    rem = d["remaining"]
    t = texts(d)
    if rem is not None and rem > 0.5:
        check(f"{acc}: guide/backup state the same amount still needed ({money(rem)})", money(rem) in t,
              "amount not found in guide/backup")
    elif d["need"] is not None:
        check(f"{acc}: plan covers the need, so no text asks for more", "still need another" not in t, t[:200])

print("\n1b) The customer message says what the confirmed plan does, not only the 'before' number")
d = S.simulate(dict(account_id=20111, keep=55, target=6, checked=True, cut=2178))
check("message: before 3.4 months, then 6.0 months with the confirmed cut, meeting the goal",
      "about 3.4 months" in d["message"] and "With the change you confirmed (cutting $2,178 a month)" in d["message"]
      and "about 6.0 months, meeting the goal of 6 months" in d["message"], d["message"])
d = S.simulate(dict(account_id=20111, keep=55, target=6, preview_cut=2178))
check("message: a preview is not described as the customer's plan", "you confirmed" not in d["message"], d["message"])

print("\n1c) A shock that ends: the target verdict follows the month-by-month balance")
d = S.simulate(dict(account_id=20111, keep=0, target=4, lasts=1, after_pct=100))
check("20111, no income for 1 month then back to 100%: lasts the 4-month target (runs out ~5.0 months)",
      d["lasts"] is True and d["short_by"] == 0 and abs(d["timed"]["out"] - 4.99) < 0.05, (d["lasts"], d["timed"]["out"]))
check("same case: the message agrees (meets the 4-month goal)", "meeting the goal of 4 months" in d["message"], d["message"])
d = S.simulate(dict(account_id=20111, keep=0, target=6, lasts=1, after_pct=100))
check("same case with a 6-month target: short, and short by the balance below $0 at month 6",
      d["lasts"] is False and d["short_by"] > 0 and d["short_by"] == -min(d["timed"]["path"][1:7]), (d["lasts"], d["short_by"]))
d = S.simulate(dict(account_id=20111, keep=55, target=6))
check("no timing set: the verdict is the judges' runway as before (3.4 < 6 → short)", d["lasts"] is False and d["timed"] is None)
r = agent.parse_rules("My income drops to zero for one month, then goes back to normal.")
check("'income drops to zero for one month, then back to normal' is read: -100%, lasts 1, then 100%",
      r["other_income_change_pct"] == -100 and r["shock_lasts_months"] == 1 and r["income_after_pct"] == 100, r)

print("\n2) Plain-English input: direction, amount and timing are read, nothing is dropped silently")
cases = [("My spending decreases by 10%.", "spending_change_pct", -10.0),
         ("They cut their spending by 10%.", "spending_change_pct", -10.0),
         ("Their spending drops 15%.", "spending_change_pct", -15.0),
         ("Their hours are cut 30% and spending goes up 5%.", "spending_change_pct", 5.0),
         ("Prices rise 4%.", "spending_change_pct", 4.0),
         ("In 3 months they lose their job; EI pays 55% for 6 months.", "starts_in_months", 3),
         ("In 3 months they lose their job; EI pays 55% for 6 months.", "shock_lasts_months", 6)]
for text, key, want in cases:
    got = agent.parse_rules(text)[key]
    check(f"{text!r} -> {key} = {want}", got == want, got)
for text in ("Interest rates rise 2 points on their variable-rate loan.", "They win a trip to Hawaii."):
    r = agent.parse_rules(text)
    check(f"{text!r} is flagged as not modelled", bool(r["unsupported"]), r)
t = S.twist(dict(account_id=20111, text="My spending decreases by 10%."))
check("the screen note shows the sign of the change", "-10" in t["note"], t["note"])

print("\n3) Correlation is never presented as the effect of an action")
EFFECT = re.compile(r"\b(?:will|would|can)\s+(?:reduce|lower|cut|decrease)\s+(?:your|the|their)?\s*risk|"
                    r"\b(?:reduces|lowers|cuts)\s+(?:your|the|their)?\s*risk\s+by|risk\s+(?:drops|falls)\s+by", re.I)
for acc in ACCOUNTS:
    d = S.simulate(dict(account_id=acc, keep=55, bills_pct=15, target=6))
    m = EFFECT.search(texts(d))
    check(f"{acc}: no 'this action reduces risk by X' wording", m is None, m and m.group(0))
check("plan ranking is labelled as association, not effect", "not a measured effect" in (HERE.parent / "core.py").read_text()
      or "not a measured effect" in (HERE / "web" / "index.html").read_text())

print("\n4) Spending not classed as fixed is not treated as cuttable")
for acc in ACCOUNTS:
    d = S.simulate(dict(account_id=acc, keep=55, bills_pct=15, target=6))
    t = texts(d)
    check(f"{acc}: no 'flexible spending' wording", "flexible spending" not in t.lower(), "found")
    check(f"{acc}: nothing counts as confirmed until the customer confirms it", d["confirmed"] == 0, d["confirmed"])
check("chart button says preview, and a separate button records the customer's confirmation",
      "Preview this cut on the chart" in JS and "Customer confirms this cut" in JS
      and "not counted in the plan until the customer confirms" in JS)
d = S.simulate(dict(account_id=20111, keep=55, target=6, preview_cut=2178))
check("a preview draws the cut on the chart but counts nothing (confirmed $0, plan not active)",
      d["confirmed"] == 0 and not d["plan_active"] and d["preview"]["display"] == "6.0" and d["remaining"] > 0,
      (d["confirmed"], d["plan_active"], d["preview"]))
d = S.simulate(dict(account_id=20111, keep=55, target=6, checked=True, cut=2178, preview_cut=999))
check("once confirmed, any preview is ignored", d["preview"] is None and d["confirmed"] == 2178, d["preview"])

print("\n4b) Single spending categories are not applied to all spending")
for text in ("My groceries increase by 10%.", "Gas prices go up 20%.", "Childcare costs rise 15%."):
    r = agent.parse_rules(text)
    check(f"{text!r}: not applied to total spending, and flagged", r["spending_change_pct"] == 0 and r["unsupported"]
          and "New monthly costs" in r["unsupported"][0], r)
r = agent.parse_rules("Prices rise 4% and groceries go up 10%.")
check("general prices still apply (+4%) while groceries are flagged", r["spending_change_pct"] == 4 and r["unsupported"], r)
t = S.twist(dict(account_id=20111, text="My groceries increase by 10%."))
check("screen: groceries card adds no monthly cost and shows the warning", t["shock"]["extraOut"] == 0 and t["warn"], t)

print("\n5) Money not yet saved is never added to today's cash")
for acc in ACCOUNTS:
    d = S.simulate(dict(account_id=acc, keep=100, one_off=3000, target=6))
    bad = [p["title"] for p in d["plans"] if p.get("apply") and set(p["apply"]) - {"cut", "label"}]
    check(f"{acc}: only monthly cuts can go on the chart", not bad, bad)
d = S.simulate(dict(account_id=20111, keep=55, target=6, plan_buffer=50000))
check("a 'plan_buffer' sent by an old page is ignored", d["planned"]["headline"] == d["after"]["headline"],
      (d["planned"]["headline"], d["after"]["headline"]))

print("\n6) AUC is shown with its population and evaluation, never as accuracy")
check("risk note explains AUC as ranking on a hold-out, not percent-correct",
      "hold-out" in JS and "not a percent-correct" in JS)
for f in ("core.py", "README.md"):
    s = (HERE.parent / f).read_text()
    s = re.sub(r"not (?:an? )?(?:percent-correct|accuracy)|\(not accuracy\)", "", s)
    m = re.search(r"AUC[^.\n]{0,60}accura|accura[^.\n]{0,30}AUC|\d\d% accura", s)
    check(f"{f}: AUC never called accuracy", m is None, m and m.group(0))

print("\n7) Live demo works with no AI and no network")
for acc, want in {20111: (3.4, 4.0, 3.4), 21786: (1.0, 0.0, 1.0), 55447: ("no", 5.0, 5.0)}.items():
    S_ = float(S.prof.loc[acc].S)
    got = [S.simulate(dict(account_id=acc, keep=55, target=6))["after"],
           S.simulate(dict(account_id=acc, keep=100, one_off=S_, target=6))["after"],
           S.simulate(dict(account_id=acc, keep=55, bills_pct=15, target=6))["after"]]
    got = [g["display"] for g in got]
    exp = {20111: ["3.4", "4.0", "3.4"], 21786: ["1.0", "0.0", "1.0"], 55447: ["5.0", "no shortfall (120+)", "5.0"]}[acc]
    check(f"{acc}: judges' examples without AI {got}", got == exp, exp)
t = S.twist(dict(account_id=20111, text="They lose their job (EI at 55%) and their household bills go up 15% in the same month."))
check("twist card read with no AI (rule-based reader)", t["source"] != "Gemini" and t["shock"]["keep"] == 55
      and t["shock"]["bills"] == 15, t)
d = S.simulate(dict(account_id=20111, keep=55, target=6))
check("customer message written with no AI (template)", d["message"].startswith("Hi"), d["message"][:60])

print(f"\nResult: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
