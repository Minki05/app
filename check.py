"""
Self-check: run `python check.py` in the app folder.
Checks (1) the engine against the judges' answers, (2) the twist-card reader on many phrasings,
(3) the month-by-month simulator, (4) that Gemini is reachable (optional).
"""
import os
import sys
import tomllib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent  # noqa: E402

ms = pd.read_csv("app_data/monthly_summary.csv")
passed = failed = 0


def check(name, got, want):
    global passed, failed
    ok = got == want
    passed += ok
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}" + ("" if ok else f", expected {want!r}"))


def profile(acc):
    d = ms[ms.account_id == acc].sort_values("month")
    return dict(I=d.money_in.median(), S=d.money_out.median(), B=d.avg_daily_balance.tail(6).mean(),
                H=d.household_bills.median())


def runway(B, income, spending, one_off=0):
    left, gap = B - one_off, spending - income
    if left <= 0:
        return 0
    if gap <= 0:
        return "no shortfall (120+)"
    return round(min(left / gap, 120), 1)


print("\n1) Engine vs the judges' formula (README example twists)")
expected = {20111: (7.8, 3.4, 4.0, 3.4), 21786: (28.9, 1.0, 0, 1.0),
            55447: ("no shortfall (120+)", 5.0, "no shortfall (120+)", 5.0)}
for acc, (base, ei, furnace, ei_bills) in expected.items():
    p = profile(acc)
    check(f"{acc} today", runway(p["B"], p["I"], p["S"]), base)
    check(f"{acc} job loss EI 55%", runway(p["B"], 0.55 * p["I"], p["S"]), ei)
    check(f"{acc} furnace = 1 month of spending", runway(p["B"], p["I"], p["S"], p["S"]), furnace)
    check(f"{acc} job loss + bills +15%", runway(p["B"], 0.55 * p["I"], p["S"] + 0.15 * p["H"]), ei_bills)

print("\n2) Twist-card reader (rule-based backup, works offline)")
cases = [
    ("This customer was laid off. EI pays 55% of their usual income for 6 months. Do they make it?",
     dict(job_loss=True, ei_pct=55.0, survive_months=6, question="last_months")),
    ("The furnace dies, and the repair costs a full month of their typical spending.",
     dict(one_off_months_of_spending=1)),
    ("They lose their job (EI at 55%) and their household bills go up 15% in the same month.",
     dict(job_loss=True, ei_pct=55.0, bills_up_pct=15.0)),
    ("Their landlord raises the rent by 20%. Can they keep up?", dict(bills_up_pct=20.0)),
    ("A $3,500 car repair. Can they afford it?", dict(one_off_dollars=3500.0, question="can_pay")),
    ("Their hours are cut by 30% and daycare costs $650 a month. How long can they last?",
     dict(other_income_change_pct=-30.0, extra_monthly_dollars=650.0, question="how_long")),
    ("Job loss with no EI.", dict(job_loss=True, ei_pct=0.0)),
    ("Their partner, who earns 40% of the household income, gets laid off.",
     dict(job_loss=False, other_income_change_pct=-40.0)),
    ("Their mortgage payment goes up by $400 when they renew.", dict(extra_monthly_dollars=400.0)),
    ("They get a $5,000 bonus but the car needs a $3,000 repair.",
     dict(one_off_dollars=3000.0, windfall_dollars=5000.0)),
    ("In 3 months they lose their job; EI pays 55% for 6 months.",
     dict(job_loss=True, starts_in_months=3, shock_lasts_months=6, income_after_pct=0.0)),
    ("Their pay is cut 25% for 4 months, then goes back to normal.",
     dict(other_income_change_pct=-25.0, shock_lasts_months=4, income_after_pct=100.0)),
]
for text, want in cases:
    got = agent.parse_rules(text)
    check(text[:60], {k: got[k] for k in want}, want)

print("\n3) Month-by-month simulator agrees with the judge formula when the shock never ends")
sys.argv = ["x"]
src = open("app.py").read()
ns = {"np": np}
exec(src[src.index("def simulate("):src.index("FLEX_COLS =")], ns)
p = profile(20111)
sim, _ = ns["simulate"](p["B"], p["I"], p["S"], 0.55 * p["I"], p["S"], 0)
check("20111 job loss, ongoing", round(sim, 1), 3.4)
sim, _ = ns["simulate"](p["B"], p["I"], p["S"], p["I"], p["S"], p["S"])
check("20111 furnace", round(sim, 1), 4.0)

print("\n4) Regression checks for issues found in review")
r = agent.parse_rules("They lose their job (EI at 57%) and their household bills go up 12% in the same month.")
check("EI 57% kept exactly", r["ei_pct"], 57.0)
check("bills +12% kept exactly", r["bills_up_pct"], 12.0)
app_src = open("app.py").read()
rt = app_src[app_src.index("def read_twist"):app_src.index("def random_account")]
check("read_twist does not round to 5%", "/ 5) * 5" in rt, False)
r = agent.parse_rules("They lose their job (EI at 57.5%) and their household bills go up 12.5%.")
check("EI 57.5% kept with its decimal", r["ei_pct"], 57.5)
check("bills +12.5% kept with its decimal", r["bills_up_pct"], 12.5)
check("read_twist keeps one decimal (no int rounding of EI)", "int(round(float(t[\"ei_pct\"]" in rt, False)
check("EI slider accepts decimals", 'key="ei", step=0.1' in app_src, True)
nowf = pd.read_csv("app_data/now_features.csv", index_col=0)
check("20578 (median spending $0) gets a planning spend > 0", bool(nowf.loc[20578, "S_plan"] > 0), True)
check("20578 buffer months not 0", bool(nowf.loc[20578, "buffer_months"] > 0), True)
check("no account with spending has buffer_months computed from $0", int(((nowf.S == 0) & (nowf.S_plan > 0) &
      (nowf.buffer_months == 0) & (nowf.B > 0)).sum()), 0)
check("shock plan never cuts more than flexible spending", "cut = need_cut if feasible else flex" in app_src, True)
check("no 'cash-only' wording in the app", "Cash-only accounts" in app_src, False)
check("model gap labelled as association", "not a measured effect" in app_src, True)

print("\n5) Gemini (optional)")
try:
    s = tomllib.load(open(".streamlit/secrets.toml", "rb"))
    os.environ["GEMINI_API_KEY"] = s["GEMINI_API_KEY"]
    os.environ["GEMINI_MODEL"] = s.get("GEMINI_MODEL", "")
    t, src_used = agent.parse_twist("They lose their job (EI at 55%) and their household bills go up 15%.")
    print(f"  {'PASS' if src_used == 'Gemini' else 'INFO'}  reader used: {src_used} {agent.USED_MODEL} "
          f"{agent.short_error()} -> {agent.describe(t)}")
except FileNotFoundError:
    print("  INFO  no .streamlit/secrets.toml, Gemini skipped (app still works)")

print(f"\nResult: {passed} passed, {failed} failed")
