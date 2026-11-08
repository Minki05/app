"""
Agent layer: reads a twist card written in plain English and writes a message to the customer.

- Gemini is used when a key is available (st.secrets["GEMINI_API_KEY"] or env GEMINI_API_KEY).
- If Gemini is missing or fails, a rule-based reader and a template take over, so the demo never stops.
- Gemini never calculates anything: all numbers come from the engine.
"""
import json
import os
import re

WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "half a": 0.5, "half": 0.5,
            "full": 1, "a full": 1}
NUM = r"(\d+(?:\.\d+)?|half a|half|a full|full|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _num(s):
    s = s.lower().strip()
    return WORD_NUM[s] if s in WORD_NUM else float(s)


def _get_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"], st.secrets.get("GEMINI_MODEL", "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.8-flash")
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL", "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.8-flash")


LAST_ERROR = ""
USED_MODEL = ""


def _new_client(genai, types, key):
    try:   # 15-second limit per model (Gemini minimum is 10s) so a busy model never freezes the live demo
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=15_000))
    except Exception:
        return genai.Client(api_key=key)


def _gemini(prompt, want_json):
    """Calls Gemini, trying each model in GEMINI_MODEL (comma-separated) until one answers.
    A fresh client is made for every call (Streamlit reruns can leave an old client closed).
    Returns text, or None (and sets LAST_ERROR) if every model fails."""
    global LAST_ERROR, USED_MODEL
    key, models = _get_key()
    if not key:
        LAST_ERROR = "no GEMINI_API_KEY in .streamlit/secrets.toml"
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        LAST_ERROR = f"google-genai not installed ({e})"
        return None
    cfg = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json" if want_json else "text/plain")
    client = _new_client(genai, types, key)               # held in a variable for the whole call
    for model in [m.strip() for m in models.split(",") if m.strip()]:
        for attempt in range(2):                           # one retry with a new client if it was closed
            try:
                r = client.models.generate_content(model=model, contents=prompt, config=cfg)
                LAST_ERROR, USED_MODEL = "", model
                return r.text
            except Exception as e:
                LAST_ERROR = f"{model}: {type(e).__name__}: {str(e)[:160]}"
                if "closed" in str(e).lower() and attempt == 0:
                    client = _new_client(genai, types, key)
                    continue
                break
    return None


def short_error():
    """One short, human reason for the last Gemini failure (for the UI)."""
    e = LAST_ERROR.lower()
    if not e:
        return ""
    if "no gemini_api_key" in e:
        return "no API key set"
    if "503" in e or "unavailable" in e or "high demand" in e:
        return "Gemini is busy right now"
    if "timeout" in e or "timed out" in e:
        return "Gemini took too long"
    if "404" in e or "not_found" in e:
        return "model name not found"
    if "401" in e or "403" in e or "api key" in e or "permission" in e:
        return "API key problem"
    if "429" in e or "quota" in e or "resource_exhausted" in e:
        return "Gemini quota reached"
    return "Gemini unavailable: " + LAST_ERROR.split(": ", 1)[-1][:70]


# ---------------------------------------------------------------------
# 1. Read a twist card
# ---------------------------------------------------------------------
EMPTY = dict(job_loss=False, ei_pct=None, other_income_change_pct=0.0, bills_up_pct=0.0,
             extra_monthly_dollars=0.0, extra_income_dollars=0.0, spending_change_pct=0.0, one_off_dollars=0.0,
             one_off_months_of_spending=0.0, windfall_dollars=0.0, starts_in_months=0, shock_lasts_months=0,
             income_after_pct=100.0, survive_months=None, question="auto", unsupported=[])
QUESTIONS = ("auto", "can_pay", "last_months", "how_long")
PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)"
MONTHLY = r"(?:/\s*mo(?:nth)?|per month|a month|each month|every month|monthly)"


_POSS = r"(?:their|his|her|the|a|your|my|our)"
JOB_LOSS = (r"laid off|lay-?off|(?:lose|lost|loses) " + _POSS + r" jobs?|"
            r"job loss|unemploy|fired|out of work|lose work")
# "did not / didn't / haven't / was not / never / no longer" right before the phrase, in the same clause
NEGATION = re.compile(r"(?:\bnot\b|n't\b|\bnever\b|\bno longer\b)(?:\s+[\w']+){0,3}\s*$")


def job_loss_mentions(text, pattern=JOB_LOSS):
    """(affirmed, negated): whether the text states a job loss, and whether it explicitly denies one.
    "I did not lose my job. My rent goes up 10%." -> (False, True): only that phrase is dropped, the rent
    rise in the next sentence is still read by the other rules."""
    t = " " + str(text).lower().replace(",", "") + " "
    affirmed = negated = False
    for m in re.finditer(pattern, t):
        clause = re.split(r"[.;:!?]|\bbut\b|\band\b", t[:m.start()])[-1]
        if NEGATION.search(clause):
            negated = True
        else:
            affirmed = True
    return affirmed, negated


def _ei_state(t, m):
    """How an EI share match is meant, from the words before it in its clause:
    'negated' ("do not receive EI at 55%"), 'hypothetical' ("if I did, EI would pay 55%") or 'real'."""
    clause = re.split(r"[.;:!?]|\bbut\b|\band\b", t[:m.start()])[-1]
    if re.search(r"\b(?:not|never|no)\b|n't\b", clause):
        return "negated"
    if re.search(r"\b(?:if|unless|whether)\b|\bin case\b", clause):
        return "hypothetical"
    return "real"


CATEGORY = (r"groceries|grocery bills?|food(?: costs?| prices?)?|gas(?: prices?)?|fuel|gasoline|transit|"
            r"phone bills?|childcare|daycare|insurance(?: premiums?)?|car insurance")


NEG = r"\b(?:not|never|no|won't|doesn't|don't|didn't|isn't|aren't|wouldn't|without)\b|n't\b"


def negated_at(t, m, back=30):
    """True when the matched change is denied ("does not increase by 10%", "no rent increase"). Only the same
    clause counts: "I did not lose my job. My rent goes up 10%" is a real rent rise."""
    before = re.split(r"[.;!?]|\b(?:and|but|while|so)\b", t[max(m.start() - back, 0):m.start()])[-1]
    return bool(re.search(NEG, before + " " + m.group(0)))


def category_warning(cat, pct):
    return (f"{cat} +{float(pct):g}%: the data doesn't show how much goes to {cat}, so this was not applied. "
            f"Enter the extra cost per month under 'New monthly costs' if you know it")


def parse_rules(text):
    """Backup reader: regular expressions for the common ways a twist card is phrased."""
    t = " " + text.lower().replace(",", "") + " "
    out = dict(EMPTY, unsupported=[])
    # --- job loss + EI   (customers write in the first person too: my / our / I / we)
    poss = _POSS
    partner = re.search(r"\b(?:partner|spouse|husband|wife)\b", t)
    affirmed, negated = job_loss_mentions(text)
    if affirmed and negated and not partner:
        # "I was not fired. I was laid off." — both a denial and a statement: don't pick one as fact
        out["unsupported"].append("job loss unclear (the card both states and denies one): not applied, "
                                  "lower the money-in setting by hand if it applies")
    elif affirmed and not partner:
        out["job_loss"] = True
    # "55% of my pay" names the share directly, so it wins; "EI ... 55%" must stay inside one sentence,
    # otherwise "... from EI. My rent goes up 15%" would read the rent rise as the EI share.
    # When a partner loses the job, "40% of our income" is the partner's share (handled below), not EI.
    m = (not partner and re.search(PCT + r"\s*of (?:their|his|her|the|your|my|our)? ?(?:usual|normal|regular|old)? ?"
                                         r"(?:income|pay|salary|wages?)", t)) or \
        re.search(r"\bei\b[^%\d.;!?]{0,40}" + PCT, t)
    ei_unreal = False                    # EI only mentioned as denied ("do not receive EI") or as a what-if
    if m:
        ei = _ei_state(t, m)
        if partner:                                   # unchanged: partner cards are read below
            out["ei_pct"], out["job_loss"] = float(m.group(1)), True
        elif negated:
            # the job loss is denied (or unclear): an EI share never turns it into one
            ei_unreal = True
            if ei == "real" and not affirmed:
                out["unsupported"].append("EI share mentioned but the card says there was no job loss: "
                                          "not applied")
        elif ei == "negated":                         # "I lost my job and do not receive EI at 55%"
            ei_unreal = True
        elif ei == "hypothetical" and not affirmed:   # "if I qualify, EI would pay 55%" with no job loss stated
            ei_unreal = True
        else:
            out["ei_pct"], out["job_loss"] = float(m.group(1)), True
    if out["job_loss"] and out["ei_pct"] is None:
        no_ei = re.search(r"(?:no|without|not eligible for|doesn't get|does not get|isn't eligible for)\s+(?:any\s+)?"
                          r"(?:ei|employment insurance)", t)
        out["ei_pct"] = 0.0 if no_ei or ei_unreal else (55.0 if re.search(r"\bei\b|employment insurance", t) else 0.0)
    # --- other income changes (pay cut, fewer hours, raise)
    m = re.search(r"(?:pay|salary|wages?|hours|income)\s+(?:is\s+|are\s+|gets?\s+)?(?:cut|drops?|falls?|reduced|goes down|go down)"
                  r"[^.%\d]{0,15}" + PCT, t) or \
        re.search(PCT + r"\s+(?:pay|salary|wage|income|hours)\s+cut", t) or \
        re.search(r"(?:cut|reduce)s?\s+(?:[\w']+\s+){0,2}(?:pay|salary|hours|income|wages?)\s+by\s+" + PCT, t)
    to_m = re.search(r"(?:pay|salary|wages?|hours|income|earnings)\s+(?:is\s+|are\s+)?(?:cut|drops?|falls?|reduced|goes down|"
                     r"go down|goes|go)\s+to\s+" + PCT, t)
    if to_m and not negated_at(t, to_m):         # "drops TO 70%" = 70% remains, a 30% drop
        out["other_income_change_pct"] = float(to_m.group(1)) - 100.0
    elif m and negated_at(t, m):
        out["unsupported"].append(f"'{m.group(0).strip()}' is stated as NOT happening, so it was not applied")
    elif m:
        out["other_income_change_pct"] = -float(m.group(1))
    elif not out["job_loss"] and re.search(r"(?:income|pay|paycheque|paycheck|earnings)s?\s+(?:drops?|falls?|goes|go|is cut)"
                                           r"\s+to\s+(?:zero|nothing|\$?0)\b|\bno income\b|\bzero income\b", t):
        out["other_income_change_pct"] = -100.0
    m = re.search(r"(?:gets? a (?:pay )?raise|pay raise|pay rise|salary (?:goes up|rises|increases)|"
                  r"income (?:goes up|rises|increases))[^.%\d]{0,15}" + PCT, t)
    if m:
        out["other_income_change_pct"] = float(m.group(1))
    if re.search(r"\b(?:partner|spouse|husband|wife)\b", t) and \
            job_loss_mentions(text, r"laid off|lay-?off|loses?\s+\w+\s+job|lost\s+\w+\s+job|fired|out of work|unemploy")[0]:
        m = re.search(r"(?:earn|earns|earned|brings? in|makes?)\s+" + PCT, t)
        share = float(m.group(1)) if m else 50.0
        if not m:
            out["unsupported"].append("partner's share of income not given: assumed half (adjust the slider)")
        ei = out["ei_pct"] if out["ei_pct"] is not None else 0.0
        # household keeps (100 - share)% plus EI on the partner's share: 60% + 40% x 55% = 82%
        out["other_income_change_pct"] = -share * (1 - ei / 100.0)
        out["job_loss"], out["ei_pct"] = False, None
        if ei:
            out["unsupported"].append(f"partner's EI ({ei:g}% of their {share:g}% share) counted: household keeps "
                                      f"{100 - share * (1 - ei / 100):g}% of usual money in")
    # --- one category of spending (groceries, gas...): the data has no per-category amounts, so a % of it can't be
    # turned into dollars. Checked BEFORE bills, so "grocery bills up 10%" is not read as household bills.
    for m in re.finditer(r"\b(" + CATEGORY + r")\b[^.%\d]{0,30}" + PCT, t):
        out["unsupported"].append(category_warning(m.group(1), m.group(2)))
        t = t.replace(m.group(0), " ")
    # --- bills / rent
    m = re.search(r"(?:bills?|rent|utilit\w*|hydro|heating)[^.%]{0,40}?(?:up|rise|rises|increase|increases|jump|jumps|"
                  r"go up|goes up|climb|climbs)[^.%\d]{0,15}" + PCT, t) or \
        re.search(PCT + r"\s*(?:higher|increase|rise|more|jump)[^.]{0,25}(?:bills?|rent|utilit)", t) or \
        re.search(r"(?:raises?|increases?|hikes?|ups?)\s+(?:the\s+|their\s+|your\s+)?(?:rent|bills?|utilit\w*)"
                  r"[^.%\d]{0,15}" + PCT, t) or \
        re.search(r"\b(?:bills?|rent|utilit\w*)\s*(?:\+|up\s+)?\s*" + PCT, t)   # shorthand: "bills 15%", "bill +15%"
    t_wo_bills = t
    if m and negated_at(t, m):
        out["unsupported"].append(f"'{m.group(0).strip()}' is stated as NOT happening, so it was not applied")
        t_wo_bills = t.replace(m.group(0), " ")
    elif m:
        out["bills_up_pct"] = float(m.group(1))
        t_wo_bills = t.replace(m.group(0), " ")
    # --- general spending change (inflation, prices)
    m = re.search(r"(?:prices?|cost of living|inflation|spending)[^.%\d]{0,30}" + PCT, t_wo_bills)
    if m and negated_at(t_wo_bills, m):
        out["unsupported"].append(f"'{m.group(0).strip()}' is stated as NOT happening, so it was not applied")
        m = None
    if m:
        # direction: "decreases / drops / falls ... by 10%" after the word, or "cut / reduce ... spending" right
        # before it, is a decrease; anything else is an increase
        before = t_wo_bills[max(m.start() - 25, 0):m.start()]
        down = re.search(r"decreas|drop|fall|fell|down|lower|reduc|cut|shrink|less|declin", m.group(0)) or \
            re.search(r"(?:cut|cuts|reduce|reduces|lower|lowers|decrease|decreases)\s+(?:[\w']+\s+){0,2}$", before)
        out["spending_change_pct"] = -float(m.group(1)) if down else float(m.group(1))
    # --- loan / mortgage payment goes up by $X
    for m in re.finditer(r"(?:loan|mortgage|car|lease|student loan|credit card)\s+payments?[^.$]{0,30}?"
                         r"(?:up|rise|rises|increase|increases|goes up|jumps?)[^.$\d]{0,10}\$\s?(\d+(?:\.\d+)?)", t):
        out["extra_monthly_dollars"] += float(m.group(1))
        t = t.replace(m.group(0), " ")
    # --- money coming IN once (bonus, refund, gift)
    for m in re.finditer(r"(?:bonus|tax refund|refund|inheritance|inherit\w*|gift|windfall|severance|payout|lottery)"
                         r"[^.$]{0,25}\$\s?(\d+(?:\.\d+)?)\s*(k|thousand)?|\$\s?(\d+(?:\.\d+)?)\s*(k|thousand)?"
                         r"\s*(?:bonus|tax refund|refund|inheritance|gift|windfall|severance|payout)", t):
        amt, k = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        out["windfall_dollars"] += float(amt) * (1000 if k else 1)
        t = t.replace(m.group(0), " ")
    # --- timing: when it starts, how long it lasts, what income looks like after
    m = re.search(r"(?:in|starting in|starts in|after|from)\s+" + NUM + r"\s+months?(?:\s+from now|\s+time)?", t)
    if m and not re.search(r"for\s+" + NUM + r"\s+months?", t[max(0, m.start() - 4):m.end()]):
        out["starts_in_months"] = int(round(_num(m.group(1))))
    ASK = r"(?:make it|keep up|survive|get through|last|stay afloat|hold on|hang on)(?:\s+through)?(?:\s+(?:for|the next|over))?"
    target = re.search(ASK + r"\s+" + NUM + r"\s+months?", t) or re.search(ASK + r"\s+(?:a|one|1|a full|the)\s+(year)", t)
    t_dur = t.replace(target.group(0), " ") if target else t
    m = re.search(r"(?:for|lasts?|over|next)\s+" + NUM + r"\s+months?", t_dur)
    if m:
        out["shock_lasts_months"] = int(round(_num(m.group(1))))
        if out["job_loss"]:
            back = re.search(r"(?:new job|back to work|re-?employed|find work|finds work|hired again)", t)
            out["income_after_pct"] = 100.0 if back else 0.0
    # --- recurring new monthly costs ($X a month)
    for m in re.finditer(r"\$\s?(\d+(?:\.\d+)?)\s*(k|thousand)?\s*" + MONTHLY, t):
        amt = float(m.group(1)) * (1000 if m.group(2) else 1)
        before = t[max(m.start() - 45, 0):m.start()]
        clause = re.split(r"[.;!?]|\band\b", before)[-1]          # only the words in the same clause
        if re.search(NEG, clause):
            out["unsupported"].append(f"'{clause.strip()} ${amt:,.0f} a month' is stated as NOT happening: not applied")
        elif re.search(r"income|pay\b|salary|wages?|earn|side (?:job|gig|hustle)|raise|benefit|child benefit|"
                       r"pension|rental income|money in", clause):
            sign = -1 if re.search(r"decreas|drop|fall|fell|down|lower|reduc|cut|less|lose|loses|lost", clause) else 1
            out["extra_income_dollars"] += sign * amt
        elif re.search(r"decreas|drop|fall|fell|down|lower|reduc|cut|less|cheaper|sav(?:e|es|ing)", clause):
            out["extra_monthly_dollars"] -= amt
        else:
            out["extra_monthly_dollars"] += amt
    t_wo_monthly = re.sub(r"\$\s?\d+(?:\.\d+)?\s*(?:k|thousand)?\s*" + MONTHLY, " ", t)
    # --- one-off costs
    m = re.search(NUM + r"\s+(?:full\s+)?months?\s*(?:'s|’s)?\s+(?:worth\s+)?of\s+(?:their\s+|his\s+|her\s+|your\s+)?"
                        r"(?:typical\s+|usual\s+|normal\s+)?(?:spending|expenses)", t)
    if m:
        out["one_off_months_of_spending"] = _num(m.group(1))
    total = 0.0
    t_wo_monthly = re.sub(r"(?:bonus|tax refund|refund|inheritance|gift|windfall|severance|payout)[^.$]{0,25}\$\s?\d+"
                          r"(?:\.\d+)?\s*(?:k|thousand)?", " ", t_wo_monthly)
    for m in re.finditer(r"\$\s?(\d+(?:\.\d+)?)\s*(k|thousand)?", t_wo_monthly):
        total += float(m.group(1)) * (1000 if m.group(2) else 1)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(k|thousand)?\s*(?:dollars|cad)\b", t_wo_monthly):
        total += float(m.group(1)) * (1000 if m.group(2) else 1)
    out["one_off_dollars"] = total
    # --- how long they need to get through: asked directly ("make it through a year"), else the shock period
    if target:
        out["survive_months"] = 12 if target.group(1) == "year" else int(round(_num(target.group(1))))
    elif out["shock_lasts_months"] and re.search(r"make it|survive|get through|keep up|stay afloat", t):
        out["survive_months"] = out["shock_lasts_months"]
    if re.search(r"interest rates?|rate (?:hike|increase|rise)|variable[- ]rate|basis points", t) and \
            not out["extra_monthly_dollars"]:
        out["unsupported"].append("interest-rate change: the data can't split loan interest from principal, "
                                  "so enter the higher monthly payment under 'New monthly costs'")
    found = any(out[k] for k in ("job_loss", "other_income_change_pct", "bills_up_pct", "spending_change_pct",
                                 "extra_monthly_dollars", "extra_income_dollars", "one_off_dollars", "one_off_months_of_spending",
                                 "windfall_dollars"))
    if not found and not out["unsupported"]:
        out["unsupported"].append("no shock the engine can model was found in this card")
    # --- what is being asked
    if re.search(r"can (?:they|he|she|you|i|we) (?:afford|pay|cover)|afford it|enough to (?:pay|cover)", t):
        out["question"] = "can_pay"
    elif re.search(r"make it|survive|get through|keep up|stay afloat|last (?:the|that|those|until)|enough to last", t):
        out["question"] = "last_months"
    elif re.search(r"how long", t):
        out["question"] = "how_long"
    return out


def _clean(d):
    """Validate and clamp whatever the model returned, so bad output can never break the engine."""
    out = dict(EMPTY, unsupported=[])
    num = lambda v, lo, hi, default=0.0: default if v is None else max(lo, min(hi, float(v)))
    out["job_loss"] = bool(d.get("job_loss", False))
    out["ei_pct"] = num(d.get("ei_pct"), 0, 100, None) if out["job_loss"] else None
    if out["job_loss"] and out["ei_pct"] is None:
        out["ei_pct"] = 0.0
    out["other_income_change_pct"] = num(d.get("other_income_change_pct"), -100, 100)
    out["bills_up_pct"] = num(d.get("bills_up_pct"), 0, 300)
    out["extra_monthly_dollars"] = num(d.get("extra_monthly_dollars"), -1e6, 1e6)
    out["extra_income_dollars"] = num(d.get("extra_income_dollars"), -1e6, 1e6)
    out["spending_change_pct"] = num(d.get("spending_change_pct"), -90, 300)
    out["one_off_dollars"] = num(d.get("one_off_dollars"), 0, 1e7)
    out["one_off_months_of_spending"] = num(d.get("one_off_months_of_spending"), 0, 24)
    out["windfall_dollars"] = num(d.get("windfall_dollars"), 0, 1e7)
    out["starts_in_months"] = int(num(d.get("starts_in_months"), 0, 24))
    out["shock_lasts_months"] = int(num(d.get("shock_lasts_months"), 0, 60))
    out["income_after_pct"] = num(d.get("income_after_pct"), 0, 150, 100.0)
    sm = d.get("survive_months")
    out["survive_months"] = None if sm in (None, 0) else int(num(sm, 1, 24))
    out["question"] = d.get("question") if d.get("question") in QUESTIONS else "auto"
    out["unsupported"] = [str(x)[:80] for x in (d.get("unsupported") or [])][:3]
    return out


PARSE_PROMPT = """You turn a financial "twist card" read out by a judge into numbers for a resilience calculator.
Return ONLY a JSON object with exactly these keys:
- job_loss: true if the customer themself loses their job, else false
- ei_pct: % of usual income paid by Employment Insurance after a job loss (e.g. 55); 0 if no EI; null if no job loss
- other_income_change_pct: any OTHER change to income in %, e.g. pay cut 20% -> -20, fewer hours 30% -> -30,
  raise 10% -> 10, partner loses a job and they earned 40% of household income -> -40 (if their share is not
  given, use -50 and add "partner's share of income assumed to be half" to unsupported);
  a higher loan/mortgage payment of $X goes in extra_monthly_dollars; 0 if none
- bills_up_pct: % increase in household bills, rent or utilities; 0 if none
- extra_monthly_dollars: change in recurring costs per month in dollars (childcare $800/month -> 800; rent decreases by $200 a month -> -200); 0 if none
- extra_income_dollars: change in recurring money IN per month in dollars (income increases by $500 a month -> 500); 0 if none
- A change the card says does NOT happen ("spending does not increase") is not applied: leave it at 0
- spending_change_pct: % change in overall spending, negative for a decrease (prices up 5% -> 5, spending decreases 10% -> -10); 0 if none. A change to ONE category only (groceries, gas, childcare...) is NOT overall spending: use 0 and add it to unsupported
- one_off_dollars: total one-time costs in dollars (repairs, bills, emergencies); 0 if none
- one_off_months_of_spending: one-time cost stated as months of typical spending (e.g. "a full month" -> 1); 0 if none
- windfall_dollars: one-time money coming IN (bonus, tax refund, gift, severance); 0 if none
- starts_in_months: months from now until the shock starts (e.g. "in 3 months" -> 3); 0 if it starts now
- shock_lasts_months: how many months the shock lasts before things change (e.g. "EI for 6 months" -> 6,
  "rent goes up for 3 months" -> 3); 0 if it is ongoing / not stated
- income_after_pct: income after the shock period ends, as % of usual. EI running out with no new job -> 0;
  temporary pay cut that ends -> 100; if not stated assume 0 after a job loss and 100 otherwise
- survive_months: months the support lasts or the period to get through (e.g. "EI for 6 months" -> 6); null if not stated
- question: "can_pay" if asking whether they can afford/pay a cost, "last_months" if asking whether they make it
  through a period, "how_long" if asking how long money lasts, otherwise "auto"
- unsupported: list of short phrases for anything in the card these fields cannot represent (else [])
Use only what the card says. Never invent numbers.

Examples:
Card: "This customer was laid off. EI pays 55% of their usual income for 6 months. Do they make it?"
{"job_loss": true, "ei_pct": 55, "other_income_change_pct": 0, "bills_up_pct": 0, "extra_monthly_dollars": 0,
 "spending_change_pct": 0, "one_off_dollars": 0, "one_off_months_of_spending": 0, "windfall_dollars": 0,
 "starts_in_months": 0, "shock_lasts_months": 6, "income_after_pct": 0, "survive_months": 6,
 "question": "last_months", "unsupported": []}
Card: "The furnace dies, and the repair costs a full month of their typical spending."
{"job_loss": false, "ei_pct": null, "other_income_change_pct": 0, "bills_up_pct": 0, "extra_monthly_dollars": 0,
 "spending_change_pct": 0, "one_off_dollars": 0, "one_off_months_of_spending": 1, "windfall_dollars": 0,
 "starts_in_months": 0, "shock_lasts_months": 0, "income_after_pct": 100, "survive_months": null,
 "question": "can_pay", "unsupported": []}
Card: "They lose their job (EI at 55%) and their household bills go up 15% in the same month."
{"job_loss": true, "ei_pct": 55, "other_income_change_pct": 0, "bills_up_pct": 15, "extra_monthly_dollars": 0,
 "spending_change_pct": 0, "one_off_dollars": 0, "one_off_months_of_spending": 0, "windfall_dollars": 0,
 "starts_in_months": 0, "shock_lasts_months": 0, "income_after_pct": 0, "survive_months": null,
 "question": "auto", "unsupported": []}
Card: "Their hours get cut by a quarter and daycare adds $650 a month. How long can they keep going?"
{"job_loss": false, "ei_pct": null, "other_income_change_pct": -25, "bills_up_pct": 0, "extra_monthly_dollars": 650,
 "spending_change_pct": 0, "one_off_dollars": 0, "one_off_months_of_spending": 0, "windfall_dollars": 0,
 "starts_in_months": 0, "shock_lasts_months": 0, "income_after_pct": 100, "survive_months": null,
 "question": "how_long", "unsupported": []}

Card: \"\"\"{card}\"\"\""""


CROSS_CHECK = ("extra_income_dollars", "other_income_change_pct", "bills_up_pct", "extra_monthly_dollars", "spending_change_pct",
               "one_off_dollars", "one_off_months_of_spending", "windfall_dollars")
ADDED = []          # what the rule check filled in on the last Gemini read (shown on screen)


def parse_twist(text):
    """Returns (params, source). Gemini reads the card; the rule-based reader then checks it, and any shock Gemini
    left at 0 that the rules clearly found is filled in (and listed in ADDED), so one missed phrase can't drop a shock."""
    global ADDED
    ADDED = []
    raw = _gemini(PARSE_PROMPT.replace("{card}", text), want_json=True)
    if raw:
        try:
            d = _clean(json.loads(raw.strip().removeprefix("```json").removesuffix("```")))
        except Exception:
            d = None
        if d is not None:
            r = parse_rules(text)
            affirmed, negated = job_loss_mentions(text)
            unclear = ("job loss unclear (the card both states and denies one): not applied, "
                       "lower the money-in setting by hand if it applies")
            if affirmed and negated and not r["job_loss"]:
                # conflicting phrases (rules didn't apply a job loss either): surface it, don't pick a side
                d["job_loss"], d["ei_pct"] = False, None
                d["unsupported"].append(unclear)
            elif r["job_loss"] and not d["job_loss"]:
                if negated:                      # conflicting phrases: surface it, don't pick a side as fact
                    d["unsupported"].append(unclear)
                else:                            # a clearly stated job loss Gemini missed
                    d["job_loss"], d["ei_pct"] = True, r["ei_pct"]
                    ADDED.append(f"job loss with EI at {r['ei_pct']:g}%")
            elif d["job_loss"] and negated and not affirmed:
                # the card explicitly denies a job loss and never states one: don't apply it as a fact
                d["job_loss"], d["ei_pct"] = False, None
                d["unsupported"].append("the card says there was no job loss, so none was applied; "
                                        "lower the money-in setting by hand if it applies")
            if re.search(r"\b(?:partner|spouse|husband|wife)\b", text.lower()) and not r["job_loss"] and \
                    r["other_income_change_pct"] and d["job_loss"]:
                d["job_loss"], d["ei_pct"] = False, None
                d["other_income_change_pct"] = r["other_income_change_pct"]
                d["unsupported"] += [u for u in r["unsupported"] if u.startswith("partner")]
            for k in ("spending_change_pct", "bills_up_pct", "other_income_change_pct"):
                if any("NOT happening" in u for u in r["unsupported"]) and not r[k] and d[k]:
                    d[k] = 0.0
            d["unsupported"] += [u for u in r["unsupported"] if "NOT happening" in u and u not in d["unsupported"]]
            cats = [m for m in re.finditer(r"\b(" + CATEGORY + r")\b[^.%\d]{0,30}" + PCT, text.lower())]
            general = re.search(r"prices?|cost of living|inflation|spending", re.sub(
                r"\b(?:" + CATEGORY + r")\b", " ", text.lower()))
            if cats and not general and d["spending_change_pct"]:
                d["spending_change_pct"] = 0.0
                for m in cats:
                    d["unsupported"].append(category_warning(m.group(1), m.group(2)))
            for k in CROSS_CHECK:
                if r[k] and not d[k]:
                    d[k] = r[k]
                    ADDED.append(describe({**EMPTY, "unsupported": [], k: r[k]}))
            return d, "Gemini"
    return parse_rules(text), "rule-based reader"


def describe(params):
    parts = []
    if params["job_loss"]:
        parts.append(f"job loss with EI at {params['ei_pct']:g}%")
    if params["other_income_change_pct"]:
        parts.append(f"income {params['other_income_change_pct']:+g}%")
    if params["bills_up_pct"]:
        parts.append(f"household bills +{params['bills_up_pct']:g}%")
    if params["spending_change_pct"]:
        parts.append(f"spending {params['spending_change_pct']:+g}%")
    if params["extra_monthly_dollars"]:
        x = params["extra_monthly_dollars"]
        parts.append(f"monthly costs {'+' if x > 0 else '-'}${abs(x):,.0f}/month")
    if params.get("extra_income_dollars"):
        x = params["extra_income_dollars"]
        parts.append(f"money in {'+' if x > 0 else '-'}${abs(x):,.0f}/month")
    if params["one_off_dollars"]:
        parts.append(f"one-off cost ${params['one_off_dollars']:,.0f}")
    if params["one_off_months_of_spending"]:
        parts.append(f"one-off cost of {params['one_off_months_of_spending']:g} month(s) of spending")
    if params.get("windfall_dollars"):
        parts.append(f"one-off money in ${params['windfall_dollars']:,.0f}")
    if params.get("starts_in_months"):
        parts.append(f"starts in {params['starts_in_months']} months")
    if params.get("shock_lasts_months"):
        parts.append(f"lasts {params['shock_lasts_months']} months, then income at {params['income_after_pct']:g}%")
    if params["survive_months"]:
        parts.append(f"{params['survive_months']} months to get through")
    q = {"can_pay": "question: can they pay?", "last_months": "question: do they make it?",
         "how_long": "question: how long does money last?"}.get(params.get("question"))
    if q:
        parts.append(q)
    return ", ".join(parts) or "no shock found"


# ---------------------------------------------------------------------
# 2. Write a message to the customer
# ---------------------------------------------------------------------
def bank_offer(bank):
    """A bank step as something to ask about, never as an approval or a promise: there is no approval
    system behind this app, so "Tangerine could X" becomes "Ask us whether we can X"."""
    bank = str(bank or "").strip()
    if bank.startswith("Tangerine could "):
        return "Ask us whether we can " + bank[len("Tangerine could "):]
    return bank


def _can_pay(facts):
    """(can pay?, cash left after paying) from the unrounded engine values when the caller passed them
    (web app); older callers (app.py) only pass the rounded display amounts."""
    left = facts["left"] if facts.get("left") is not None else facts["buffer"] - facts["one_off"]
    can = facts["can_pay"] if facts.get("can_pay") is not None else left >= 0
    return bool(can), float(left)


def _dollars(x):
    """Whole-dollar display; amounts under $1 are said as such so a rounded "$0" never hides a real gap."""
    return "less than $1" if 0 < abs(x) < 0.5 else f"${abs(x):,.0f}"


def template_message(facts):
    lines = [f"Hi there! We looked at how your account would handle {facts['shock']}."]
    r = facts["runway_after"]
    if facts.get("one_off_only"):
        can, left = _can_pay(facts)
        if can and left > 0:
            lines.append(f"You can pay the ${facts['one_off']:,} cost from savings and still have {_dollars(left)} left.")
        elif can:
            lines.append(f"You can pay the ${facts['one_off']:,} cost from savings, but it would use all of it "
                         f"and leave no cash cushion.")
        else:
            lines.append(f"Your savings (${facts['buffer']:,}) would not cover the ${facts['one_off']:,} cost "
                         f"({_dollars(left)} short).")
    elif isinstance(r, str):
        lines.append("Good news: your money coming in would still cover your usual spending.")
    elif facts.get("meets_target") is None:
        lines.append(f"Your savings would last about {_months(facts)} months, "
                     f"{'meeting' if _meets(facts) else 'short of'} the goal of {facts['survive']} months.")
    else:
        lines.append(f"If the change never ended, your savings would last about {r} months.")
    mbm = facts.get("month_by_month")
    if mbm is not None:
        goal = ("meeting" if facts.get("meets_target") else "short of") + f" the goal of {facts['survive']} months"
        lines.append(f"Counting when the shock starts and ends, your money runs out in about {mbm} months, {goal}."
                     if not isinstance(mbm, str) else
                     f"Counting when the shock starts and ends, your money does not run out, {goal}.")
    plan_line = _plan_line(facts)
    if plan_line:
        lines.append(plan_line)
    if facts["steps"]:
        lines.append("Here is what we suggest, in order:")
        for i, (step, bank) in enumerate(facts["steps"][:3], 1):
            lines.append(f"{i}. {step} {bank_offer(bank)}")
    lines.append("We're here to help you set any of this up.")
    return "\n".join(lines)


def _meets(facts):
    """The same verdict the screen shows (unrounded), never re-derived from a rounded number."""
    return facts["meets"] if facts.get("meets") is not None else facts["runway_after"] >= facts["survive"]


def _months(facts):
    """Rounded months, unless rounding would hide the verdict (5.96 shown as '6.0' but short of 6): then 2 decimals."""
    r, raw = facts["runway_after"], facts.get("runway_after_raw")
    if raw is not None and not _meets(facts) and r >= facts["survive"]:
        return f"{raw:.2f}"
    return r


def _plan_line(facts):
    """The result with the changes the customer confirmed (never a preview)."""
    pl = facts.get("plan")
    if not pl:
        return ""
    what = " and ".join(x for x in (f"cutting ${pl['cut']:,} a month" if pl["cut"] else "",
                                    f"${pl['extra']:,} a month extra coming in" if pl["extra"] else "") if x)
    r = str(pl["runway"])
    lasts = ("money coming in would cover your spending" if r.startswith("no shortfall")
             else f"your savings would last about {r.split()[0]} months")
    goal = "meeting" if pl["meets"] else "still short of"
    return f"With the change you confirmed ({what}), {lasts}, {goal} the goal of {facts['survive']} months."


def write_message(facts):
    """Returns (message, source)."""
    r = facts["runway_after"]
    if facts.get("one_off_only"):
        can, left = _can_pay(facts)
        if can and left > 0:
            outlook = (f"The one-off cost of ${facts['one_off']:,} can be paid from savings, "
                       f"leaving {_dollars(left)}.")
        elif can:
            outlook = (f"The one-off cost of ${facts['one_off']:,} can be paid from savings, but it uses all of "
                       f"the savings: no cash cushion is left afterwards.")
        else:
            outlook = (f"Savings of ${facts['buffer']:,} are not enough for the ${facts['one_off']:,} cost "
                       f"({_dollars(left)} short).")
    elif isinstance(r, str):
        outlook = "After the shock, money coming in still covers usual spending, so savings are not used up."
    elif facts.get("meets_target") is not None:
        outlook = f"If the change never ended, savings would last about {r} months."
    else:
        verdict = "which meets" if _meets(facts) else "which is shorter than"
        outlook = (f"After the shock, savings would last about {_months(facts)} months, {verdict} "
                   f"the goal of lasting {facts['survive']} months.")
    now = facts["runway_now"]
    today = ("Today, money coming in covers usual spending." if isinstance(now, str) else
             "Today, savings are already used up." if not now else
             f"Today, with no shock, savings would last about {now} months.")
    clean = {
        "shock": facts["shock"],
        "today": today,
        "outlook_after_shock": outlook,
        "steps_in_order": [f"{step} {bank_offer(bank)}" for step, bank in facts["steps"][:4]],
    }
    if facts.get("month_by_month") is not None:
        goal = ("which meets" if facts.get("meets_target") else "which is shorter than") + \
            f" the goal of lasting {facts['survive']} months"
        clean["month_by_month_note"] = (f"Counting when the shock starts and ends, money runs out in about "
                                        f"{facts['month_by_month']} months, {goal}"
                                        if not isinstance(facts["month_by_month"], str)
                                        else f"Counting when the shock starts and ends, money does not run out, {goal}")
    if _plan_line(facts):
        clean["with_the_plan_the_customer_confirmed"] = _plan_line(facts)
    prompt = f"""You are a friendly, plain-spoken banking assistant writing to one customer.
Write a short message (max 120 words). Start with "Hi there," (never use an account number).
Rules:
- Use ONLY the facts below. Do not invent, round or change any number.
- Write in everyday words: never copy technical labels such as "no shortfall", "120+" or "runway".
- If a confirmed-plan result is given, say it right after the outlook (before -> with the plan).
- State the outlook exactly as given; do not say the customer "can survive" the goal if the outlook says it is shorter.
- Keep every condition in a step (for example "Once the monthly deficit is closed").
- Bank support is only something to ask about: never say the bank has approved, pre-approved, will provide or
  can definitely offer anything, and never say the customer qualifies.
- No percentages or statistics about other customers. Do not blame the customer. Be supportive.
- Put the steps in a short numbered list, in the same order.
Facts (JSON): {json.dumps(clean, default=str)}"""
    raw = _gemini(prompt, want_json=False)
    if raw and raw.strip():
        return raw.strip(), "Gemini"
    return template_message(facts), "template"
