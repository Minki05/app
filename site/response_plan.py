"""Scenario response rules, not customer risk scores or learned clusters."""
from dataclasses import dataclass
import math
import engine as E

LABELS = {'A': 'Immediate cash need', 'B': 'No monthly shortfall',
          'C': 'Target period covered', 'D': 'Monthly adjustment needed',
          'X': 'Data check needed'}

@dataclass(frozen=True)
class Response:
    group: str
    required: float | None
    end_buffer: float | None
    reason: str
    action: str


def classify(buffer, gap, target):
    """Use unrounded amounts; no imputation and no arbitrary risk thresholds."""
    m = lambda x: f'${abs(x):,.2f}'
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (buffer, gap, target)) or target <= 0:
        return Response('X', None, None, 'The balance, monthly cash flow or target is missing or invalid.',
                        'Check the source figures before using a plan.')
    if buffer <= 0:
        reason = (f'The model starts {m(buffer)} below zero.' if buffer < 0 else
                  'The model starts with no cushion to draw on; there is no initial negative balance.')
        action = (f'First review how to cover the initial {m(buffer)} gap; check whether payment timing or instalments are available.'
                  if buffer < 0 else 'First review how to establish a cash cushion and cover upcoming payments.')
        return Response('A', None, None, reason, action)
    required = E.improvement_needed(gap, buffer, target)
    end = buffer - gap * target
    if gap <= 0:
        return Response('B', required, end,
                        f'Monthly money in covers money out, leaving {m(-gap)} under this scenario.',
                        ('Review an affordable emergency-fund contribution; it is not automatically committed.' if gap < 0 else
                         'There is no monthly surplus to assign to saving; review funding before setting a contribution.'))
    if buffer / gap >= target:
        return Response('C', required, end,
                        f'The cushion covers {target:g} months, with {m(end)} projected to remain.',
                        f'Review the ongoing {m(gap)} monthly drawdown before the target period ends.')
    return Response('D', required, end,
                    f'The cushion lasts {buffer / gap:.2f} months against a {target:g}-month target.',
                    f'Review {m(required)} of monthly improvement, then enter only changes you consider feasible.')


def invalid_accounts(ms):
    """Prevent pandas skip-NA aggregation from hiding incomplete source figures."""
    cols = ['money_in', 'money_out', 'household_bills', 'insurance', 'loan_payments', 'avg_daily_balance']
    import numpy as np
    import pandas as pd
    values = ms[cols].apply(pd.to_numeric, errors='coerce')
    bad = (~np.isfinite(values)).any(axis=1) | (values[cols[:-1]] < 0).any(axis=1)
    bad |= ms.duplicated(['account_id', 'month'], keep=False)
    bad |= pd.to_datetime(ms['month'], format='%Y-%m', errors='coerce').isna()
    return set(ms.loc[bad, 'account_id'])
