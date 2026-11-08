"""
Resilience Rewired - Tangerine track, Sheridan Datathon 2026
Run:  streamlit run app.py     (needs app_data/ from precompute.py)
"""
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

import agent

st.set_page_config(page_title="Resilience Engine", page_icon="🍊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root { --bg:#0B0E14; --panel:#131722; --panel2:#171C28; --line:#232A37; --text:#E6E8EE; --muted:#8A93A6;
        --dim:#5B6476; --up:#16C784; --down:#EA3943; --warn:#F5C451; --cyan:#00D1FF; }
html, body, [class*="css"], .stApp, button, input, textarea { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }
.stApp { background: radial-gradient(1200px 500px at 70% -10%, rgba(0,209,255,.08), transparent 60%),
                     radial-gradient(900px 400px at 0% 0%, rgba(22,199,132,.06), transparent 60%), var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1040px; padding-top: 2rem; }
section[data-testid="stSidebar"] { background: #0F131C; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] h2 { font-size: 12px; color: var(--muted); font-weight: 700; letter-spacing: 1.2px;
                                      text-transform: uppercase; margin-top: 10px; }
.num, .mid, .big, .hero b, .stat { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace !important; }
.card { background: linear-gradient(180deg, var(--panel2), var(--panel)); border: 1px solid var(--line);
        border-radius: 14px; padding: 20px 22px; margin-bottom: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.35); }
.hero-card { padding: 26px 26px; border-color: #2A3344; }
.eyebrow { color: var(--cyan); font-size: 12px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 10px; }
.hero { color: var(--text); font-size: 28px; font-weight: 800; line-height: 1.3; letter-spacing: -0.3px; }
.hero span { text-shadow: 0 0 18px currentColor; }
.sub { color: var(--muted); font-size: 14px; margin-top: 10px; }
.sub b { color: var(--text); }
.label { color: var(--muted); font-size: 12px; font-weight: 600; letter-spacing: .3px; }
.verdict { display:inline-block; margin-top:14px; padding:6px 12px; border-radius:8px; font-size:14px; font-weight:700; }
.verdict.ok { background: rgba(22,199,132,.12); color: var(--up); border: 1px solid rgba(22,199,132,.35); }
.verdict.no { background: rgba(234,57,67,.12); color: var(--down); border: 1px solid rgba(234,57,67,.35); }
.tag { display:inline-block; margin-left:5px; padding:0 5px; border-radius:4px; background:#1E2533; color:var(--cyan);
       font-size:10px; font-weight:700; font-family:"JetBrains Mono", monospace; }
.group { color: var(--dim); font-size: 11px; font-weight: 700; letter-spacing: 1px; margin: 16px 0 8px; }
.grid3 { display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.ci { color: var(--dim); font-size: 12px; }
.stat { color: var(--text); font-weight: 700; }
.label.strong { color: var(--text); font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px; }
.big { font-size: 36px; font-weight: 700; margin: 6px 0 12px; text-shadow: 0 0 22px currentColor; }
.mid { color: var(--text); font-size: 19px; font-weight: 700; margin-top: 4px; }
.tiny { color: var(--dim); font-size: 12px; margin-top: 10px; }
.bar { height: 6px; background: #1E2533; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; box-shadow: 0 0 12px currentColor; }
.grid5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.do b { color: var(--text); font-family: "JetBrains Mono", monospace; }
.fgrid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 760px) { .fgrid { grid-template-columns: 1fr; } }
.fnum { font-family:"JetBrains Mono", monospace; font-size: 26px; font-weight: 700; margin: 4px 0 10px; }
.frow { display:grid; grid-template-columns: 150px 1fr 64px; gap: 8px; align-items:center; font-size:12px;
        color:#AEB5C2; margin: 5px 0; }
.ftrack { height: 8px; background:#1E2533; border-radius: 99px; overflow:hidden; }
.ftrack > div { height:100%; border-radius:99px; }
.fval { font-family:"JetBrains Mono", monospace; text-align:right; color: var(--text); }
.fnote { color: var(--dim); font-size: 12px; margin-top: 8px; line-height: 1.5; }
@media (max-width: 760px) { .grid5, .grid3 { grid-template-columns: repeat(2, 1fr); } .hero { font-size: 22px; } }
.st-key-chartbox { background: linear-gradient(180deg, var(--panel2), var(--panel)); border: 1px solid var(--line);
                   border-radius: 14px; padding: 16px 18px 6px; margin-bottom: 12px; }
.section { font-size: 13px; font-weight: 800; color: var(--text); letter-spacing: 1.4px; text-transform: uppercase; margin: 22px 4px 10px; }
.section .count { color: var(--up); margin-left: 6px; font-family: "JetBrains Mono", monospace; }
.pill { display: inline-block; font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 6px; margin-bottom: 10px;
        letter-spacing: .4px; font-family: "JetBrains Mono", monospace; }
.title { font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.do { font-size: 15px; color: #C9CED8; line-height: 1.6; margin-bottom: 14px; }
.why { font-size: 14px; color: #AEB5C2; line-height: 1.6; padding: 12px 0; border-top: 1px solid var(--line); }
.why > span:first-child { display: block; color: var(--dim); font-size: 11px; font-weight: 700; letter-spacing: 1px;
                          text-transform: uppercase; margin-bottom: 4px; }
.model { margin-top: 8px; background: rgba(0,209,255,.06); border: 1px solid rgba(0,209,255,.2); border-radius: 8px;
         padding: 10px 14px; font-size: 14px; color: #C9CED8; }
div[data-testid="stExpander"] { background: var(--panel); border-radius: 12px; border: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

A = "app_data"


# ---------------------------------------------------------------------
# Data + model (loaded once)
# ---------------------------------------------------------------------
@st.cache_data
def load():
    ms = pd.read_csv(f"{A}/monthly_summary.csv")
    train = pd.read_csv(f"{A}/train_features.csv", index_col=0)
    now = pd.read_csv(f"{A}/now_features.csv", index_col=0)
    loan_ev = json.load(open(f"{A}/loan_evidence.json"))
    try:
        ohlc = pd.read_csv(f"{A}/monthly_ohlc.csv")
    except FileNotFoundError:
        ohlc = None
    try:
        findings = json.load(open(f"{A}/findings.json"))
    except FileNotFoundError:
        findings = None
    return ms, train, now, loan_ev, ohlc, findings


# The engine (risk model, evidence, ranked plans, significance tests) lives in core.py, shared with the
# web version in site/, so both screens always show the same numbers.
from core import (AUC, AUC_CI, ACCOUNTS, BLUE, BUFFER_BANDS, GRAY, MODEL, RED, RUNWAY_BANDS, Y,  # noqa: E402
                  band, findings_data, flexible_mix, fmt_rw, model_matrix, peer, profile, recommend, risk,
                  runway, simulate)

ms, train, now, loan_ev, ohlc, findings = load()


# ---------------------------------------------------------------------
# Sidebar: customer + shock
# ---------------------------------------------------------------------
PRESETS = {
    "Custom": None,
    "Job loss (EI 55%)": dict(job=True, ei=55.0, bills=0.0, one_mode="None", one_amt=0, one_m=0.0, survive=6,
                              q_type="last_months"),
    "Furnace dies (1 month of spending)": dict(job=False, ei=55.0, bills=0.0, one_mode="Months of spending",
                                               one_amt=0, one_m=1.0, q_type="can_pay"),
    "Job loss + bills +15%": dict(job=True, ei=55.0, bills=15.0, one_mode="None", one_amt=0, one_m=0.0,
                                  q_type="last_months"),
}
SHOCK_RESET = dict(inc_other=0.0, extra=0, spend_pct=0.0, windfall=0, start_in=0, lasts=0, after_inc=100.0, twist_warn="")
for k, v in dict(job=True, ei=55.0, bills=0.0, one_mode="None", one_amt=0, one_m=0.0, acc=20111, survive=6,
                 inc_other=0.0, extra=0, spend_pct=0.0, q_type="auto", twist_warn="",
                 windfall=0, start_in=0, lasts=0, after_inc=100.0,
                 twist_text="", twist_note="", view="Advisor").items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PRESETS[st.session_state.preset]
    if p:
        for k, v in {**SHOCK_RESET, **p}.items():
            st.session_state[k] = v


def read_twist():
    text = st.session_state.twist_text.strip()
    if not text:
        return
    t, src = agent.parse_twist(text)
    st.session_state.inc_other = float(max(-100, min(50, round(float(t["other_income_change_pct"]), 1))))
    st.session_state.extra = int(round(float(t["extra_monthly_dollars"])))
    st.session_state.spend_pct = float(max(-50, min(100, round(float(t["spending_change_pct"]), 1))))
    st.session_state.q_type = t.get("question", "auto")
    st.session_state.windfall = int(round(float(t.get("windfall_dollars", 0))))
    st.session_state.start_in = int(min(max(int(t.get("starts_in_months", 0)), 0), 12))
    st.session_state.lasts = int(min(max(int(t.get("shock_lasts_months", 0)), 0), 24))
    st.session_state.after_inc = float(min(max(round(float(t.get("income_after_pct", 100)), 1), 0), 150))
    warn = list(t.get("unsupported") or [])
    if float(t["bills_up_pct"] or 0) > 500:
        warn.append(f"bills +{t['bills_up_pct']:g}% capped at +500%")
    st.session_state.twist_warn = ("Not modelled: " + "; ".join(warn)) if warn else ""
    st.session_state.job = bool(t["job_loss"])
    if t["job_loss"]:
        st.session_state.ei = float(round(float(t["ei_pct"] or 0), 1))             # keep the exact % on the card
    st.session_state.bills = float(min(round(float(t["bills_up_pct"] or 0), 1), 500))
    if t["one_off_dollars"] and t["one_off_months_of_spending"]:      # both given: keep the dollar total,
        S_now = profile(int(st.session_state.acc))[0]["S"]             # converting the months part to dollars
        t["one_off_dollars"] += t["one_off_months_of_spending"] * S_now
    if t["one_off_dollars"]:
        st.session_state.one_mode, st.session_state.one_amt = "Dollar amount", int(t["one_off_dollars"])
    elif t["one_off_months_of_spending"]:
        st.session_state.one_mode = "Months of spending"
        st.session_state.one_m = float(min(round(float(t["one_off_months_of_spending"]), 2), 6))
    else:
        st.session_state.one_mode = "None"
    # months to get through: from the card if stated, otherwise back to the 6-month default
    st.session_state.survive = int(min(max(int(t["survive_months"]), 1), 12)) if t["survive_months"] else 6
    st.session_state.preset = "Custom"
    src_label = f"Gemini ({agent.USED_MODEL})" if src == "Gemini" else src
    st.session_state.twist_note = f"Read by {src_label}: {agent.describe(t)}"
    if src != "Gemini" and agent.LAST_ERROR:
        st.session_state.twist_note += f" · backup reader used ({agent.short_error()})"


def random_account():
    st.session_state.acc = int(np.random.choice(ACCOUNTS))


with st.sidebar:
    st.markdown("### Resilience Engine")
    st.header("Customer")
    st.number_input("account_id", min_value=int(min(ACCOUNTS)), max_value=int(max(ACCOUNTS)), step=1, key="acc")
    st.button("🎲 Random customer", on_click=random_account, width="stretch")

    st.segmented_control("View", ["Advisor", "Customer"], key="view")

    st.header("Shock")
    st.text_area("Paste the twist card", key="twist_text", height=90,
                 placeholder="e.g. They lose their job (EI at 55%) and their household bills go up 15%.")
    st.button("✨ Read twist card", on_click=read_twist, width="stretch", type="primary")
    if st.session_state.twist_note:
        st.caption(st.session_state.twist_note)
    if st.session_state.twist_warn:
        st.warning(st.session_state.twist_warn, icon="⚠️")
    st.selectbox("Or pick a preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.toggle("Job loss", key="job")
    st.slider("EI pays (% of usual income)", 0.0, 100.0, key="ei", step=0.1, format="%.1f%%", disabled=not st.session_state.job)
    st.slider("Household bills increase (%)", 0.0, 500.0, key="bills", step=0.1, format="%.1f%%")
    st.radio("One-off cost", ["None", "Dollar amount", "Months of spending"], key="one_mode", horizontal=True)
    if st.session_state.one_mode == "Dollar amount":
        st.number_input("One-off cost ($)", min_value=0, step=500, key="one_amt")
    elif st.session_state.one_mode == "Months of spending":
        st.slider("One-off cost (months of typical spending)", 0.0, 6.0, step=0.05, key="one_m")
    more = any(st.session_state[k] for k in ("inc_other", "extra", "spend_pct", "windfall"))
    with st.expander("More shock types", expanded=bool(more)):
        st.slider("Other income change (%)", -100.0, 50.0, step=0.1, format="%.1f%%", key="inc_other",
                  help="Pay cut, fewer hours, a partner's lost income (negative) or a raise (positive).")
        st.number_input("New monthly costs ($/month)", min_value=0, step=50, key="extra",
                        help="New recurring costs, e.g. childcare or a car payment.")
        st.slider("Overall spending change (%)", -50.0, 100.0, step=0.1, format="%.1f%%", key="spend_pct",
                  help="e.g. prices rise 5% → +5")
        st.number_input("One-off money in ($)", min_value=0, step=500, key="windfall",
                        help="Bonus, tax refund, gift or severance that arrives once.")
    timed = bool(st.session_state.start_in or st.session_state.lasts)
    with st.expander("Timing (month-by-month view)", expanded=timed):
        st.slider("Shock starts in (months)", 0, 12, key="start_in")
        st.slider("Shock lasts (months, 0 = ongoing)", 0, 24, key="lasts",
                  help="e.g. EI paid for 6 months → 6")
        st.slider("Income after it ends (% of usual)", 0.0, 150.0, step=0.1, format="%.1f%%", key="after_inc",
                  help="EI runs out with no new job → 0. Pay cut ends → 100.")
    st.selectbox("Question on the card", ["auto", "can_pay", "last_months", "how_long"], key="q_type",
                 format_func={"auto": "Auto-detect", "can_pay": "Can they pay it?",
                              "last_months": "Do they make it through the period?",
                              "how_long": "How long does the money last?"}.get)

    st.header("Plan settings")
    survive = st.slider("Months the customer must survive", 1, 12, key="survive")
    target = st.slider("Emergency fund target (months of spending)", 1, 6, 3)

acc = int(st.session_state.acc)
if acc not in now.index:
    st.error(f"account_id {acc} not found. Try one of: {', '.join(map(str, ACCOUNTS[:5]))} …")
    st.stop()

p, hist = profile(acc)
ei_pct = (st.session_state.ei / 100 if st.session_state.job else 1.0) * (1 + st.session_state.inc_other / 100)
bills_up = st.session_state.bills / 100
extra_total = float(st.session_state.extra) + st.session_state.spend_pct / 100 * p["S"]
one_off = {"None": 0.0, "Dollar amount": float(st.session_state.one_amt),
           "Months of spending": st.session_state.one_m * p["S"]}[st.session_state.one_mode]

windfall = float(st.session_state.windfall)
one_off_cost = one_off                         # what has to be paid
one_off = one_off_cost - windfall              # net hit to savings (negative = money in)
plans, base_risk, r_shock = recommend(acc, p, ei_pct, bills_up, max(one_off, 0.0) if one_off > 0 else one_off,
                                      survive, target, extra_total)
r_now = runway(p["B"], p["I"], p["S"])

# ---------------------------------------------------------------------
# Main page (clean fintech style)
# ---------------------------------------------------------------------


def money(x):
    return f"${x:,.0f}"


def card(html, cls="card"):
    st.html(f'<div class="{cls}">{html}</div>')


def bold_numbers(text):
    """Bold dollar amounts, percentages and month counts so the action is easy to scan."""
    return re.sub(r"(\$[\d,]+(?:/month)?|\d+(?:\.\d+)?%|about \d+ months)", r"<b>\1</b>", text)


def fmt_evidence(text):
    """'12.0% (79/660, 95% CI 9.7%–14.7%)' -> bold rate + small grey count/CI."""
    return re.sub(r"(\d+\.\d%) \((\d+/\d+), 95% CI ([^)]+)\)",
                  r'<span class="stat">\1</span> <span class="ci">(\2 · CI \3)</span>', text)


shock_desc = []
if st.session_state.job:
    shock_desc.append(f"job loss (EI {st.session_state.ei:g}%)")
if st.session_state.inc_other:
    shock_desc.append(f"income {st.session_state.inc_other:+g}%")
if bills_up:
    shock_desc.append(f"bills +{st.session_state.bills:g}%")
if st.session_state.spend_pct:
    shock_desc.append(f"spending {st.session_state.spend_pct:+g}%")
if st.session_state.extra:
    shock_desc.append(f"new costs {money(st.session_state.extra)}/month")
if one_off_cost:
    shock_desc.append(f"one-off cost {money(one_off_cost)}")
if windfall:
    shock_desc.append(f"one-off money in {money(windfall)}")
if st.session_state.start_in:
    shock_desc.append(f"starts in {st.session_state.start_in} months")
if st.session_state.lasts:
    shock_desc.append(f"lasts {st.session_state.lasts} months")
shock_text = ", ".join(shock_desc) or "no shock"

# Month-by-month view (timing: delay, temporary shock, income after it ends)
inc2, sp2 = p["I"] * ei_pct, p["S"] + bills_up * p["H"] + extra_total
sim_args = dict(start_in=st.session_state.start_in, lasts=st.session_state.lasts,
                after_frac=st.session_state.after_inc / 100)
r_real, path_real = simulate(p["B"], p["I"], p["S"], inc2, sp2, one_off, **sim_args)
timing_html = ""
if st.session_state.start_in or st.session_state.lasts:
    ends = (f"after {st.session_state.lasts} months income goes to {st.session_state.after_inc:g}% of usual"
            if st.session_state.lasts else "ongoing")
    real_txt = f"money runs out in <b>{r_real:.1f} months</b>" if r_real is not None else \
               "money does <b>not</b> run out within 10 years"
    timing_html = (f'<div class="sub" style="margin-top:6px">📅 Month by month '
                   f'(starts in {st.session_state.start_in} months, {ends}): {real_txt}. '
                   f'<span class="ci">Judge formula assumes the shock never ends.</span></div>')

# Hero: the one sentence that matters
who = f"Customer {acc}" if st.session_state.view == "Advisor" else "You"
one_off_only = ei_pct >= 1.0 and bills_up == 0 and extra_total == 0 and one_off_cost > 0
question = st.session_state.q_type
if question == "auto":
    question = "can_pay" if one_off_only else "last_months"
if question == "can_pay" and one_off_cost <= 0:
    question = "last_months"
cant = "can't" if who == "You" else "can't"
if r_shock == 0:
    hero = f'{who} <span style="color:{RED}">{cant} cover this</span> from savings'
elif isinstance(r_shock, str):
    hero = f'{who} <span style="color:{BLUE}">won\'t run short</span> after this shock'
else:
    color = RED if r_shock < survive else BLUE
    hero = f'{who} can last <span style="color:{color}">{r_shock} months</span> after this shock'
if question == "can_pay":          # e.g. a repair: the question is "can they pay it?"
    have = p["B"] + windfall
    left = have - one_off_cost
    src_txt = "savings" + (f" + the {money(windfall)} coming in" if windfall else "")
    if left > 0:
        verdict = (f'<div class="verdict ok">✅ Can pay the {money(one_off_cost)} cost from {src_txt} · '
                   f'{money(left)} left</div>')
    else:
        verdict = (f'<div class="verdict no">❌ {src_txt.capitalize()} ({money(have)}) can\'t cover the '
                   f'{money(one_off_cost)} cost · {money(-left)} short</div>')
elif question == "how_long":
    verdict = (f'<div class="verdict ok">⏱ Money lasts {fmt_rw(r_shock)}</div>' if not isinstance(r_shock, str)
               else '<div class="verdict ok">⏱ Money does not run out</div>')
elif isinstance(r_shock, str) or r_shock >= survive:
    verdict = f'<div class="verdict ok">✅ Makes it through the {survive}-month target</div>'
else:
    verdict = (f'<div class="verdict no">❌ Falls short of the {survive}-month target by '
               f'{survive - r_shock:.1f} months</div>')
card(f"""
  <div class="eyebrow">{shock_text[:1].upper() + shock_text[1:]}</div>
  <div class="hero">{hero}</div>
  {verdict}
  <div class="sub">Today: {fmt_rw(r_now)} of runway &nbsp;·&nbsp; Buffer {money(p['B'])}
  ({now.loc[acc].buffer_months:.1f} months of spending)</div>
  {timing_html}
""", "card hero-card")

ADVISOR = st.session_state.view == "Advisor"

# Two side-by-side cards: risk + money (customer view shows only the money)
if ADVISOR:
    c1, c2 = st.columns([1, 2])
    with c1:
        w = min(base_risk / 0.4, 1) * 100
        rc = RED if base_risk >= 0.10 else ("#F5C451" if base_risk >= 0.04 else BLUE)
        card(f"""
          <div class="label">Chance of going below $0 within a year</div>
          <div class="big" style="color:{rc}">{base_risk:.1%}</div>
          <div class="bar"><div style="width:{w:.0f}%;background:{rc};color:{rc}"></div></div>
          <div class="tiny">Logistic model · AUC {AUC:.2f} on unseen customers</div>
        """)
else:
    c2 = st.container()
with c2:
    def cell(name, tag, v):
        t = f'<span class="tag">{tag}</span>' if ADVISOR else ""
        return f'<div><div class="label">{name}{t}</div><div class="mid">{money(v)}</div></div>'
    row1 = cell("Money in", "I", p["I"]) + cell("Spending", "S", p["S"]) + cell("Buffer (savings)", "B", p["B"])
    row2 = (cell("Household bills", "H", p["H"]) + cell("All fixed costs", "F", p["F"])
            + cell("Loan payment", "P", now.loc[acc].P))
    card(f'<div class="label strong">{"Typical month" if ADVISOR else "Your typical month"}</div>'
         f'<div class="grid3">{row1}</div>'
         f'<div class="group">MUST-PAY COSTS (included in spending)</div><div class="grid3">{row2}</div>'
         + (f'<div class="tiny">No household bills are paid from this account, so a '
            f'"bills go up" shock does not change the runway.</div>' if p["H"] == 0 else ""))

# Chart: monthly balance candles + neon projection after the shock
horizon = min(max(13, st.session_state.start_in + st.session_state.lasts + 4), 25)
future = pd.period_range(pd.Period(hist.month.iloc[-1], "M"), periods=horizon, freq="M").astype(str)
fig = go.Figure()
candles = None if ohlc is None else ohlc[ohlc.account_id == acc].sort_values("month").tail(36)
if candles is not None and len(candles):
    fig.add_trace(go.Candlestick(x=candles.month, open=candles.open, high=candles.high, low=candles.low,
                                 close=candles.close, name="Balance (monthly)",
                                 increasing=dict(line=dict(color=BLUE, width=1), fillcolor=BLUE),
                                 decreasing=dict(line=dict(color=RED, width=1), fillcolor=RED)))
else:
    fig.add_trace(go.Scatter(x=hist.month, y=hist.avg_daily_balance, name="Balance so far",
                             line=dict(color="#8A93A6", width=2)))


def neon(x, y, color, name, dash=None):
    """A glowing line: a wide transparent stroke under a thin bright one."""
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=f"rgba{rgb + (0.18,)}", width=10),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2.5, dash=dash)))


neon(future, path_real[:horizon], RED, "If nothing changes", dash="dot")
cut = next((pl.get("cut") for pl in plans if "cut" in pl), None)
if cut:
    _, path_cut = simulate(p["B"], p["I"], p["S"], inc2, sp2, one_off, cut=cut, **sim_args)
    neon(future, path_cut[:horizon], BLUE, "With the shock plan")
fig.add_hline(y=0, line_color="#5B6476", line_width=1, line_dash="dash")
fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                  paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                  legend=dict(orientation="h", y=1.1, x=0, font=dict(color="#AEB5C2")),
                  font=dict(family="JetBrains Mono, ui-monospace, monospace", color="#8A93A6", size=11),
                  xaxis=dict(showgrid=False, rangeslider=dict(visible=False), showspikes=True, spikecolor="#5B6476",
                             spikethickness=1, spikedash="dot", linecolor="#232A37"),
                  yaxis=dict(showgrid=True, gridcolor="#1C2230", tickprefix="$", zeroline=False, side="right",
                             showspikes=True, spikecolor="#5B6476", spikethickness=1, spikedash="dot"))
with st.container(border=False, key="chartbox"):
    st.html('<div class="label strong" style="padding:4px 4px 0">Balance · monthly candles &amp; projection</div>')
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# Plans
st.html(f'<div class="section">Your plan <span class="count">{len(plans)}</span></div>')
if not plans:
    card('<div class="title">You\'re in good shape</div><div class="do">This customer can absorb this shock '
         'without changes.</div>')
for i, pl in enumerate(plans, 1):
    t, bg, fg = pl["pill"]
    if ADVISOR:
        model = ""
        if pl["new"] is not None:
            model = (f'<div class="model">Model: predicted risk for this profile <b>{pl["start"]:.1%}</b>, '
                     f'for customers already in the "after" profile <b style="color:{BLUE}">{pl["new"]:.1%}</b>. '
                     f'<br><span class="ci">{pl.get("test", "")} · Association in the data, not a measured effect of the action.</span></div>')
        card(f"""
          <div class="pill" style="background:{bg};color:{fg}">{t}</div>
          <div class="title">{pl['title']}</div>
          <div class="do">{bold_numbers(pl['do'])}</div>
          <div class="why"><span>For this customer</span>{pl['you']}</div>
          <div class="why"><span>In the data</span>{fmt_evidence(pl['data'])}</div>
          <div class="why"><span>What Tangerine can do</span>{pl['bank']}</div>
          {model}
        """)
    else:
        card(f"""
          <div class="pill" style="background:{bg};color:{fg}">{t}</div>
          <div class="title">{pl['title']}</div>
          <div class="do">{bold_numbers(pl['do'])}</div>
          <div class="why"><span>Why this matters for you</span>{pl['you_c']}</div>
          <div class="model">🤝 {pl['bank']}</div>
        """)

# Message to the customer (agent)
facts = dict(customer=acc, shock=shock_text, runway_now=r_now, runway_after=r_shock, survive=survive,
             one_off_only=bool(question == "can_pay"), one_off=round(one_off_cost), buffer=round(p["B"] + windfall),
             month_by_month=(None if not (st.session_state.start_in or st.session_state.lasts)
                             else (round(r_real, 1) if r_real is not None else "does not run out")),
             steps=[(pl["do"], pl["bank"]) for pl in plans])
key = (acc, shock_text, survive, target)
if st.session_state.get("msg_key") != key:
    st.session_state.msg_key, st.session_state.msg_src = key, "template"
    st.session_state.msg = agent.template_message(facts)
st.html('<div class="section">Message to the customer</div>')
card(f'<div class="do" style="white-space:pre-line;margin:0">{st.session_state.msg}</div>'
     f'<div class="tiny">Written by {st.session_state.msg_src}. Numbers come from the engine, not the AI.</div>')
if st.button("✨ Rewrite with Gemini", key="gem_msg"):
    m, src = agent.write_message(facts)
    st.session_state.msg, st.session_state.msg_src = m, src
    if src != "Gemini":
        st.toast(f"{agent.short_error()}: kept the template message.")
    st.rerun()

if not ADVISOR:
    st.stop()

# ---------------------------------------------------------------------
# What the data shows: six findings, all computed from past data -> what happened next
# ---------------------------------------------------------------------
def bars(rows, scale=None, fmt="{:.1%}"):
    """rows: (label, value, color, note). Horizontal bars on one scale."""
    top = scale or max(v for _, v, _, _ in rows) or 1
    out = ""
    for label, v, color, note in rows:
        out += (f'<div class="frow"><div>{label} <span class="ci">{note}</span></div>'
                f'<div class="ftrack"><div style="width:{min(v / top, 1) * 100:.0f}%;background:{color};'
                f'box-shadow:0 0 10px {color}"></div></div><div class="fval">{fmt.format(v)}</div></div>')
    return out


def rate(mask):
    return float(Y[mask].mean()), int(Y[mask].sum()), int(mask.sum())


f_cards = []
# 1. No transfers with other banks
r1, k1, n1 = rate(train.no_other_bank == 1)
r0, k0, n0 = rate(train.no_other_bank == 0)
html = bars([("No transfers w/ other banks", r1, RED, f"{k1}/{n1}"), ("Has transfers", r0, BLUE, f"{k0}/{n0}")])
if findings:
    lr = {int(a): (b, c) for a, b, c in findings["loan_no_other_bank"]}
    orr = findings["loan_or"]
    html += ('<div class="fnote">Same pattern for loan default (pre-loan data): '
             f'{lr[1][0] / lr[1][1]:.1%} vs {lr[0][0] / lr[0][1]:.1%}, odds ratio {orr[0]:.1f} '
             f'(95% CI {orr[1]:.1f}–{orr[2]:.1f}) controlling for savings vs loan payment.</div>')
html += ('<div class="fnote">Means no direct-deposit pay or bill transfers show up. Not "cash only": '
         'many still use a card or pay bills at the bank.</div>')
f_cards.append(("1 · No transfers with other banks", f"{r1 / r0:.0f}x", "more likely to go below $0 within a year", html))
# 2. Savings cliff
bands = [(-99, 1, "Under 1 month"), (1, 2, "1–2 months"), (2, 3, "2–3 months"), (3, 6, "3–6 months"), (6, 99, "6+ months")]
rows = []
for lo, hi, lab in bands:
    r, k, n_ = rate((train.buffer_months >= lo) & (train.buffer_months < hi))
    rows.append((lab, r, RED if lo < 1 else ("#F5C451" if lo < 3 else BLUE), f"{k}/{n_}"))
f_cards.append(("2 · Under one month of savings is a cliff", f"{rows[0][1]:.0%}", "went below $0 within a year",
                bars(rows) + '<div class="fnote">Savings measured in months of the customer\'s own typical spending.</div>'))
# 3. Savings vs loan payment
if findings:
    rows = [(f"{lab} payments", k / n_, RED if lab == "<4" else ("#F5C451" if lab == "4-8" else BLUE), f"{k}/{n_}")
            for lab, k, n_ in findings["loan_coverage"]]
    f_cards.append(("3 · Savings vs loan payment predicts default", f"{rows[0][1]:.0%}",
                    "of borrowers with savings under 4 payments defaulted",
                    bars(rows) + '<div class="fnote">Savings before the loan ÷ monthly loan payment; outcome = later default.</div>'))
# 4. Overdrafts repeat
ra, ka, na = rate(train.went_negative_before == 1)
rb, kb, nb = rate(train.went_negative_before == 0)
note4 = ""
if findings:
    k, n_ = findings["loan_neg_before"]
    note4 = (f'<div class="fnote">All {k} of {n_} borrowers with a negative balance before their loan later defaulted '
             f'(small group: a warning sign, not a rule).</div>')
f_cards.append(("4 · Overdrafts repeat", f"{ra / rb:.0f}x", "more likely to go negative again",
                bars([("Went negative before", ra, RED, f"{ka}/{na}"), ("Never", rb, BLUE, f"{kb}/{nb}")]) + note4))
# 5. Who you are vs how money moves
if findings:
    rows = [(k, v, "#5B6476", "") for k, v in findings["who_auc"].items()] + \
           [(k, v, BLUE, "") for k, v in findings["how_auc"].items()]
    f_cards.append(("5 · How money moves beats who you are", "≈ 0.5", "AUC for age, gender, region (0.5 = coin flip)",
                    bars(rows, scale=1.0, fmt="{:.2f}") + '<div class="fnote">Each variable alone, predicting going '
                    'below $0 next year. Grey = who the customer is; green = account behaviour.</div>'))
# 6. Judge runway works; our model adds to it
rows = []
for lo, hi, lab in [(0, 3, "Under 3 months"), (3, 6, "3–6 months"), (6, 12, "6–12 months"), (12, 121, "12+ months")]:
    r, k, n_ = rate((train.job_loss_runway >= lo) & (train.job_loss_runway < hi))
    rows.append((lab, r, RED if lo < 3 else ("#F5C451" if lo < 6 else BLUE), f"{k}/{n_}"))
f_cards.append(("6 · The judges' runway works, our model adds to it", f"AUC {AUC:.2f}",
                "risk model on held-out accounts",
                bars(rows) + '<div class="fnote">Bars: job-loss runway (judge formula) vs going below $0 next year. '
                'The model adds savings, transfers and past overdrafts.</div>'))

tests = [c.get("test", "") for c in findings_data()["cards"]]
if len(tests) == len(f_cards):
    f_cards = [(t, big, sub, body + (f'<div class="fnote" style="color:#C9CED8">🧪 {tst}</div>' if tst else ""))
               for (t, big, sub, body), tst in zip(f_cards, tests)]
st.html('<div class="section">What the data shows <span class="count">6</span></div>')
st.html('<div class="fnote" style="margin:-4px 4px 12px">Features from each account\'s past; outcome = what happened '
        f'next (3,660 accounts, next 12 months; 682 loans). Reproduced in discovery.py.</div>')
grid = "".join(f'<div class="card" style="margin:0"><div class="label strong" style="margin-bottom:6px">{t}</div>'
               f'<div class="fnum" style="color:{RED if i < 4 else BLUE}">{big}</div>'
               f'<div class="label" style="margin:-6px 0 10px">{sub}</div>{body}</div>'
               for i, (t, big, sub, body) in enumerate(f_cards))
st.html(f'<div class="fgrid">{grid}</div>')

with st.expander("How the numbers are calculated"):
    st.markdown("""
- **Money in (I)** median of `money_in` · **Fixed costs (F)** median of household bills + insurance + loan payments
- **Spending (S)** median of `money_out` · **Buffer (B)** mean of `avg_daily_balance` over the last 6 months
- **Runway** = (B − one-off cost) ÷ (spending after shock − income after shock), capped at 120 months
- **In the data** = what actually happened to similar customers: features from each account's past, outcome = balance
  below zero in the following 12 months (3,660 accounts with 24+ months of history), 95% Wilson CIs.
- **Risk model** = logistic regression on buffer (months of spending), no transfers with other banks, and past negative
  balance. Plans are ranked by the gap in *predicted* risk between this profile and the "after" profile: an
  association in the data, not a measured effect of the action.
""")
with st.expander("Model details"):
    ci = np.exp(MODEL.conf_int())
    st.dataframe(pd.DataFrame({"odds ratio": np.exp(MODEL.params), "95% CI low": ci[0], "95% CI high": ci[1],
                               "p-value": MODEL.pvalues}).round(3))
    st.caption(f"AUC on 30% held-out accounts: {AUC:.3f} (bootstrap 95% CI {AUC_CI[0]:.2f}–{AUC_CI[1]:.2f}). "
               f"Data: PKDD'99 Financial dataset (CTU Prague).")
