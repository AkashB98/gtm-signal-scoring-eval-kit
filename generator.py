"""Deterministic synthetic GTM account / activity generator.

SYNTHETIC DATA ONLY — the vendor "Meridian Metrics" and every account
company here is fictional, invented for this project. No real account,
customer or CRM data is used anywhere.

Shape of the planted world (seeded variant, the default):

* 600 fictional B2B accounts, 26 weeks (6 months) of activity.
* Exactly 90 accounts convert in the fit window (weeks 8-12) and
  exactly 90 convert in the held-out window (weeks 20-25); 420 never
  convert. Conversion timestamps are ground truth.
* Five signals are *planted* through the activity, never written
  directly into a score:
    - product_usage_spike, pricing_page_visits, champion_engagement:
      honest signals — elevated firing probability for accounts that
      convert in the same window, in both windows.
    - webinar_attendance: PLANTED DECAY — strong lift in the fit
      window, firing probability collapses to baseline in held-out.
    - contract_doc_download: PLANTED LEAKAGE — for converted accounts
      the event is generated 2 days *after* conversion, so a naive
      in-sample read makes it look near-perfect.
* No honest/decay signal event is ever generated after an account's
  conversion, and no signal activity at all is generated in a window
  that starts after the account converted. That is what makes the
  leakage check an exact test rather than a judgement call.

The "clean" variant removes both plants: webinar lift is stable
across windows and contract downloads happen *before* conversion
with a moderate, honest lift. ``check`` exits 0 on clean data.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Dict, List

VENDOR = "Meridian Metrics"
BASE_DATE = date(2026, 1, 5)  # a Monday
DEFAULT_SEED = 42
N_ACCOUNTS = 600
N_FIT_CONVERTERS = 90
N_HELDOUT_CONVERTERS = 90
TOTAL_DAYS = 182  # 26 weeks, days 0..181

# Window name -> (first day, last day), both inclusive.
WINDOWS = {"fit": (0, 90), "heldout": (91, 181)}
WINDOW_WEEKS = {"fit": (0, 12), "heldout": (13, 25)}

SIGNAL_NAMES = [
    "product_usage_spike",
    "pricing_page_visits",
    "champion_engagement",
    "webinar_attendance",
    "contract_doc_download",
]
HONEST_SIGNALS = SIGNAL_NAMES[:3]
DECAY_SIGNAL = "webinar_attendance"
LEAKAGE_SIGNAL = "contract_doc_download"

# Firing probabilities: (converter-in-window, everyone else in cohort)
_PROBS = {
    "product_usage_spike": (0.62, 0.14),
    "pricing_page_visits": (0.58, 0.13),
    "champion_engagement": (0.66, 0.16),
}
_WEBINAR_FIT = (0.72, 0.10)
_WEBINAR_HELDOUT_SEEDED = (0.11, 0.10)
_WEBINAR_HELDOUT_CLEAN = (0.72, 0.10)
_CONTRACT_SEEDED = (0.92, 0.02)
_CONTRACT_CLEAN = (0.55, 0.10)

_FIRST = ["North", "Blue", "Silver", "Cedar", "Harbor", "Summit",
          "Quartz", "Willow", "Iron", "Golden", "Maple", "Coral",
          "Amber", "Juniper", "Onyx", "Prairie", "Cobalt", "Meadow",
          "Granite", "Lumen", "Verdant", "Aster", "Birch", "Cinder",
          "Delta", "Elm", "Flint", "Grove", "Haven", "Indigo"]
_SECOND = ["peak", "stone", "field", "bridge", "haven", "ridge",
           "line", "crest", "point", "shore", "gate", "brook",
           "spire", "vale", "mark", "haven Labs", "loop", "path",
           "works", "scale"]
# _SECOND entries must yield unique names when paired with _FIRST:
# 30 firsts x 20 seconds = 600 unique bases.
_SECOND = ["peak", "stone", "field", "bridge", "haven", "ridge",
           "line", "crest", "point", "shore", "gate", "brook",
           "spire", "vale", "mark", "loop", "path", "works",
           "scale", "forge"]
_SUFFIX = ["Labs", "Systems", "Analytics", "Software", "Partners"]
_INDUSTRIES = ["fintech", "healthtech", "logistics", "retail",
               "manufacturing", "education", "energy", "legal"]


def day_to_date(day: int) -> str:
    return (BASE_DATE + timedelta(days=day)).isoformat()


def _company_name(index: int) -> str:
    first = _FIRST[index // len(_SECOND)]
    second = _SECOND[index % len(_SECOND)]
    suffix = _SUFFIX[index % len(_SUFFIX)]
    return f"{first}{second.capitalize()} {suffix}"


def _event(account_id: str, event_type: str, day: int) -> Dict:
    return {"account_id": account_id, "event_type": event_type,
            "day": day, "date": day_to_date(day), "week": day // 7}


def _prob_for(signal: str, window: str, converts: bool,
              variant: str):
    if signal in _PROBS:
        pair = _PROBS[signal]
    elif signal == DECAY_SIGNAL:
        if window == "fit":
            pair = _WEBINAR_FIT
        elif variant == "clean":
            pair = _WEBINAR_HELDOUT_CLEAN
        else:
            pair = _WEBINAR_HELDOUT_SEEDED
    else:  # contract_doc_download
        pair = _CONTRACT_CLEAN if variant == "clean" else _CONTRACT_SEEDED
    return pair[0] if converts else pair[1]


def _signal_day(rng: random.Random, window: str, conv_day,
                converts_in_window: bool) -> int:
    """A pre-conversion day inside ``window`` for a signal event."""
    start, end = WINDOWS[window]
    if converts_in_window:
        return rng.randint(start, conv_day - 1)
    return rng.randint(start, end)


def generate_dataset(seed: int = DEFAULT_SEED,
                     variant: str = "seeded") -> Dict:
    if variant not in ("seeded", "clean"):
        raise ValueError(f"unknown variant: {variant!r}")
    rng = random.Random(seed)

    order = list(range(N_ACCOUNTS))
    rng.shuffle(order)
    fit_ids = {f"A{i + 1:03d}" for i in order[:N_FIT_CONVERTERS]}
    held_ids = {f"A{i + 1:03d}" for i in
                order[N_FIT_CONVERTERS:
                      N_FIT_CONVERTERS + N_HELDOUT_CONVERTERS]}

    accounts: List[Dict] = []
    events: List[Dict] = []
    weekly_activity: List[Dict] = []
    planted_firing: Dict[str, Dict] = {}
    truth_accounts: Dict[str, Dict] = {}

    for idx in range(N_ACCOUNTS):
        account_id = f"A{idx + 1:03d}"
        if account_id in fit_ids:
            conv_week = rng.randint(8, 12)
            conv_day = conv_week * 7 + 3
            conv_window = "fit"
        elif account_id in held_ids:
            conv_week = rng.randint(20, 25)
            conv_day = conv_week * 7 + 3
            conv_window = "heldout"
        else:
            conv_week = None
            conv_day = None
            conv_window = None
        account = {
            "account_id": account_id,
            "company": _company_name(idx),
            "industry": _INDUSTRIES[idx % len(_INDUSTRIES)],
            "employees": 20 + (idx * 37) % 480,
            "converted": conv_day is not None,
            "conversion_day": conv_day,
            "conversion_date": (day_to_date(conv_day)
                                if conv_day is not None else None),
            "conversion_week": conv_week,
        }
        accounts.append(account)
        truth_accounts[account_id] = {
            "converted": conv_day is not None,
            "conversion_day": conv_day,
            "conversion_week": conv_week,
            "conversion_window": conv_window,
        }
        planted_firing[account_id] = {}

        spike_weeks: Dict[str, int] = {}
        for window in ("fit", "heldout"):
            w_start, w_end = WINDOWS[window]
            already = conv_day is not None and conv_day < w_start
            converts = conv_day is not None and w_start <= conv_day <= w_end
            firing: Dict[str, bool] = {}
            if already:
                for signal in SIGNAL_NAMES:
                    firing[signal] = False
            else:
                for signal in SIGNAL_NAMES:
                    prob = _prob_for(signal, window, converts, variant)
                    firing[signal] = rng.random() < prob
                # Emit the activity the signals are computed from.
                if firing["product_usage_spike"]:
                    if converts:
                        spike_weeks[window] = rng.randint(
                            WINDOW_WEEKS[window][0], conv_week - 1)
                    else:
                        spike_weeks[window] = rng.randint(
                            *WINDOW_WEEKS[window])
                # pricing page views: >=3 fires, <=2 never does
                n_pricing = (rng.randint(3, 5) if firing["pricing_page_visits"]
                             else rng.randint(0, 2))
                for _ in range(n_pricing):
                    events.append(_event(
                        account_id, "pricing_page_view",
                        _signal_day(rng, window, conv_day, converts)))
                # champion engagement: >=2 fires, <=1 never does
                n_champ = (rng.randint(2, 4) if firing["champion_engagement"]
                           else rng.randint(0, 1))
                for _ in range(n_champ):
                    events.append(_event(
                        account_id, "champion_engagement",
                        _signal_day(rng, window, conv_day, converts)))
                # webinar attendance: >=1 fires
                if firing["webinar_attendance"]:
                    for _ in range(rng.randint(1, 2)):
                        events.append(_event(
                            account_id, "webinar_attendance",
                            _signal_day(rng, window, conv_day, converts)))
                # contract doc download: >=1 fires; the seeded plant
                # puts it 2 days AFTER conversion for converters.
                if firing["contract_doc_download"]:
                    if converts and variant == "seeded":
                        day = conv_day + 2
                    elif converts or conv_day is None:
                        day = _signal_day(rng, window, conv_day, converts)
                    else:
                        # future converter, earlier window: any day in
                        # the window precedes its conversion.
                        day = rng.randint(w_start, w_end)
                    events.append(_event(account_id,
                                         "contract_doc_download", day))
            planted_firing[account_id][window] = firing

        for week in range(26):
            window = "fit" if week <= 12 else "heldout"
            sessions = rng.randint(0, 5)
            if spike_weeks.get(window) == week:
                sessions = rng.randint(7, 9)
            weekly_activity.append({
                "account_id": account_id, "week": week,
                "sessions": sessions,
                "active_users": rng.randint(1, 6),
            })

    events.sort(key=lambda e: (e["day"], e["account_id"], e["event_type"]))
    ground_truth = {
        "accounts": truth_accounts,
        "planted_firing": planted_firing,
        "planted_patterns": (
            {"decay_signal": DECAY_SIGNAL, "leakage_signal": LEAKAGE_SIGNAL}
            if variant == "seeded" else
            {"decay_signal": None, "leakage_signal": None}),
        "counts": {"accounts": N_ACCOUNTS,
                   "fit_converters": N_FIT_CONVERTERS,
                   "heldout_converters": N_HELDOUT_CONVERTERS,
                   "never_converters": (N_ACCOUNTS - N_FIT_CONVERTERS
                                        - N_HELDOUT_CONVERTERS)},
    }
    return {
        "meta": {"vendor": VENDOR, "seed": seed, "variant": variant,
                 "base_date": BASE_DATE.isoformat(),
                 "windows": {k: list(v) for k, v in WINDOWS.items()},
                 "synthetic": True},
        "accounts": accounts,
        "weekly_activity": weekly_activity,
        "events": events,
        "ground_truth": ground_truth,
    }


def dataset_to_json(dataset: Dict) -> str:
    return json.dumps(dataset, sort_keys=True) + "\n"


def ground_truth_to_json(dataset: Dict) -> str:
    return json.dumps(dataset["ground_truth"], sort_keys=True) + "\n"
