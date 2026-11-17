"""Input validation: dataset shape and /score request shape.

Validators raise ValueError with a specific message on bad input and
never raise on a well-formed input. The CLI maps ValueError to exit
code 2; the HTTP layer maps it to a 400.
"""

from __future__ import annotations

from typing import Dict

from generator import SIGNAL_NAMES

_DATASET_KEYS = ("meta", "accounts", "weekly_activity", "events",
                 "ground_truth")
_ACCOUNT_KEYS = ("account_id", "company", "converted", "conversion_day")


def validate_dataset(obj) -> bool:
    if not isinstance(obj, dict):
        raise ValueError("dataset must be a JSON object")
    for key in _DATASET_KEYS:
        if key not in obj:
            raise ValueError(f"dataset missing key: {key}")
    accounts = obj["accounts"]
    if not isinstance(accounts, list) or not accounts:
        raise ValueError("dataset accounts must be a non-empty list")
    seen = set()
    for account in accounts:
        if not isinstance(account, dict):
            raise ValueError("each account must be an object")
        for key in _ACCOUNT_KEYS:
            if key not in account:
                raise ValueError(f"account missing key: {key}")
        if account["account_id"] in seen:
            raise ValueError(
                f"duplicate account_id: {account['account_id']}")
        seen.add(account["account_id"])
        conv = account["conversion_day"]
        if conv is not None and not (isinstance(conv, int)
                                     and not isinstance(conv, bool)
                                     and 0 <= conv <= 181):
            raise ValueError("conversion_day must be null or an int "
                             "in 0..181")
        if bool(account["converted"]) != (conv is not None):
            raise ValueError("converted flag disagrees with "
                             "conversion_day")
    if not isinstance(obj["events"], list):
        raise ValueError("dataset events must be a list")
    for event in obj["events"]:
        if not isinstance(event, dict) or "account_id" not in event \
                or "event_type" not in event or "day" not in event:
            raise ValueError("each event needs account_id, event_type, day")
        if event["account_id"] not in seen:
            raise ValueError(f"event for unknown account: "
                             f"{event['account_id']}")
    if not isinstance(obj["weekly_activity"], list) \
            or not obj["weekly_activity"]:
        raise ValueError("weekly_activity must be a non-empty list")
    return True


def validate_score_request(obj) -> Dict[str, bool]:
    """Normalise a /score body to {signal: bool} or raise ValueError."""
    if not isinstance(obj, dict):
        raise ValueError("body must be a JSON object")
    signals = obj.get("signals", obj)
    if not isinstance(signals, dict):
        raise ValueError("body must contain a 'signals' object")
    unknown = sorted(set(signals) - set(SIGNAL_NAMES))
    if unknown:
        raise ValueError(f"unknown signals: {unknown}")
    missing = [s for s in SIGNAL_NAMES if s not in signals]
    if missing:
        raise ValueError(f"missing signals: {missing}")
    for signal, value in signals.items():
        if not isinstance(value, bool):
            raise ValueError(f"signal {signal} must be a boolean")
    return {s: bool(signals[s]) for s in SIGNAL_NAMES}
