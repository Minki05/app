"""Planning Guide text for My Plan.

Builds short English sentences from numbers that are already computed elsewhere:
the response state comes from response_plan.classify (before the customer's plan),
the runways from engine.runway, the fund from engine.emergency_fund. Nothing here
re-classifies or re-calculates; it only words the results for this one account.
No LLM calls. Never fills a missing value, never uses group averages.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import engine as E
import response_plan as RP


@dataclass
class Guide:
    group: str
    label: str
    why: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)   # priority order, at most 3
    result: list[str] = field(default_factory=list)    # effect of the customer's own inputs
    open_gap: bool = False                              # something is still unresolved


def _months(r: E.RunwayResult, target: float) -> str:
    """Runway wording; two decimals when one decimal would round up onto the target."""
    if r.status == "buffer_gone":
        return "0 months"
    if r.status == "no_shortfall":
        return "no monthly shortfall"
    if r.status == "capped":
        return "120+ months"
    if r.raw < target <= E.round1(r.raw):          # rounding would hide that it falls short
        for d in (2, 3, 4):
            if float(f"{r.raw:.{d}f}") < target:
                return f"{r.raw:.{d}f} months"
        return f"just under {target:g} months"
    return f"{r.display} months"


def build(response: RP.Response, after: E.RunwayResult, planned: E.RunwayResult, target: int,
          one_off: float, confirmed: float, extra: float, fund: E.FundPlan, save: float, kmon: int,
          m, fund_after: E.FundPlan | None = None) -> Guide:
    """m formats money. `confirmed` = applied cut (0 unless checked) + extra money in.

    fund       — the emergency-fund section's goal, from the cushion before this scenario.
    fund_after — the same goal amount, but starting from the cushion left after this scenario
                 (E.emergency_fund(S, after.buffer_left, kmon, save)). Used for scenario wording.
    """
    if fund_after is None:
        fund_after = fund
    g = Guide(response.group, RP.LABELS[response.group])
    L, gap = after.buffer_left, after.gap
    extra_note = f" (includes {m(extra)} of expected extra money in, which isn't guaranteed)" if extra > 0 else ""

    if response.group == "X":
        g.why = [response.reason]
        g.actions = [response.action]
        g.open_gap = True
        return g

    if response.group == "A":                                     # no cushion to start with
        if L < 0:
            g.why.append(f"The model starts {m(-L)} below zero"
                         + (f" after the {m(one_off)} one-time expense." if one_off > 0 else "."))
            g.actions.append(f"Find {m(-L)} to cover the starting gap first.")
        else:
            g.why.append(("The one-time expense uses up the whole cushion: the model starts at $0. "
                          if one_off > 0 else "The model starts at $0. ")
                         + "There's no cushion, but no negative balance either.")
            g.actions.append("Set aside cash for upcoming payments first — there's nothing to absorb a surprise.")
        if gap > 0:
            g.why.append(f"Separately, money out is {m(gap)} a month more than money in.")
        elif gap < 0:
            g.why.append(f"Separately, money in is {m(-gap)} a month more than money out.")
        else:
            g.why.append("Separately, money in and money out are even each month.")
        if one_off > 0:
            g.actions.append(f"Ask whether the {m(one_off)} expense can be paid later or in instalments — "
                             "this may not be possible.")
        if gap > 0:
            g.actions.append(f"Then close the {m(gap)} monthly shortfall with changes you've checked.")
        g.result.append(f"Monthly changes don't cover the starting gap — it stays at {m(-L)}." if L < 0
                        else "Monthly changes don't create a starting cushion on their own.")
        if gap > 0:
            if confirmed <= 0:
                g.result.append(f"Monthly shortfall still open: {m(gap)}.")
            elif planned.gap > 0:
                g.result.append(f"With your changes ({m(confirmed)}/month){extra_note}, "
                                f"{m(planned.gap)} a month is still short.")
            else:
                g.result.append(f"Your changes ({m(confirmed)}/month){extra_note} cover the monthly shortfall.")
        g.open_gap = True
        return g

    if response.group == "B":                                     # no monthly shortfall
        surplus = -gap
        g.why.append(f"Money in is {m(surplus)} a month more than money out under this scenario." if surplus > 0
                     else "Money in and money out are even — no monthly shortfall, and no surplus either.")
        # in this group only a one-time expense makes the cushion differ from the one before the scenario
        basis = (" (the emergency-fund section below starts from the cushion before this scenario)"
                 if one_off > 0 else "")
        if fund_after.reached:
            g.actions.append(f"After this scenario the cushion ({m(L)}) still meets the {kmon}-month "
                             f"emergency-fund goal ({m(fund_after.target)}).")
        else:
            txt = (f"After this scenario the cushion ({m(L)}) is {m(fund_after.to_build)} short of the "
                   f"{kmon}-month emergency-fund goal ({m(fund_after.target)})")
            if fund.reached:
                txt += " — it met the goal before the one-time expense"
            g.actions.append(txt + basis + ".")
            if surplus <= 0:
                g.actions.append("There's no monthly surplus to save from here — a contribution would need "
                                 "money from elsewhere.")
            elif save <= 0:
                g.actions.append(f"Choose an amount to set aside. The {m(surplus)} surplus isn't committed "
                                 "automatically.")
            elif save > surplus:
                g.actions.append(f"Setting aside {m(save)} is more than the {m(surplus)} monthly surplus in this "
                                 f"scenario — the difference would need money from elsewhere. Only if it does "
                                 f"would the goal take {fund_after.months} months.")
            else:
                g.actions.append(f"Setting aside {m(save)} a month fits within the {m(surplus)} surplus and "
                                 f"reaches the goal in {fund_after.months} months from this scenario's cushion.")
        g.actions.append("No shortfall now isn't a guarantee — recheck if money in or bills change.")
        if confirmed > 0:
            g.result.append(f"There's no shortfall to cover, so your changes ({m(confirmed)}/month){extra_note} "
                            "aren't needed for this scenario.")
        return g

    if response.group == "C":                                     # target covered, cushion still shrinking
        exact = math.isclose(L / gap, target, rel_tol=0.0, abs_tol=1e-9)   # unrounded, not the display value
        if exact:
            g.why.append(f"The cushion lasts exactly {target:g} months — it meets your {target}-month target, "
                         f"with $0 left at month {target}.")
        else:
            g.why.append(f"The cushion lasts {_months(after, target)} — it extends beyond your {target}-month target, "
                         f"with about {m(response.end_buffer)} left at month {target}.")
        g.why.append(f"It still shrinks by {m(gap)} a month.")
        g.actions.append(f"To last {target} months, nothing more is needed"
                         + (" — but there's no margin left at the end." if exact else ".")
                         + f" To stop the cushion shrinking, the monthly gap of {m(gap)} would have to close.")
        g.actions.append(f"Recheck before month {target} if this situation continues past then.")
        if one_off > 0:
            g.actions.append(f"The {m(one_off)} one-time expense is already taken from the cushion here.")
        if confirmed <= 0:
            g.result.append(f"No changes entered — the cushion keeps shrinking by {m(gap)} a month.")
        elif planned.gap > 0:
            g.result.append(f"With your changes ({m(confirmed)}/month){extra_note}, it shrinks by "
                            f"{m(planned.gap)} a month instead.")
        else:
            g.result.append(f"Your changes ({m(confirmed)}/month){extra_note} cover the {m(gap)} monthly gap — "
                            "the cushion stops shrinking.")
        return g

    # D — target not covered
    req = response.required
    g.why.append(f"The cushion lasts {_months(after, target)} — short of your {target}-month target.")
    g.why.append(f"Lasting {target} months needs {m(req)} a month more (less spending or more money in).")
    g.actions.append(f"Close {m(req)} a month to last {target} months.")
    g.actions.append("Enter only cuts or extra money in you've checked — unchecked cuts count as $0. "
                     "The records can't show which spending could be cut.")
    g.actions.append(f"If that isn't reachable, plan for outside help before month {int(after.raw) + 1}.")
    remaining = max(req - confirmed, 0)
    if confirmed <= 0:
        g.result.append(f"Nothing confirmed yet — {m(req)} a month is still needed.")
        g.open_gap = True
    elif remaining > 0:
        g.result.append(f"With your changes ({m(confirmed)}/month){extra_note}: lasts {_months(planned, target)} — "
                        f"{m(remaining)} a month still needed. Target not met yet.")
        g.open_gap = True
    else:
        g.result.append(f"With your changes ({m(confirmed)}/month){extra_note}, the cushion lasts at least "
                        f"{target} months.")
    return g
