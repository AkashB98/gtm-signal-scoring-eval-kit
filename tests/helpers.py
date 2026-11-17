"""Shared cached datasets for the test suite (generation is fast but
600 accounts x several modules adds up; cache once per process)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_dataset  # noqa: E402

_CACHE = {}


def seeded():
    if "seeded" not in _CACHE:
        _CACHE["seeded"] = generate_dataset()
    return _CACHE["seeded"]


def clean():
    if "clean" not in _CACHE:
        _CACHE["clean"] = generate_dataset(variant="clean")
    return _CACHE["clean"]


def tiny_dataset(accounts, weekly=None, events=None):
    """Hand-built minimal dataset with the generator's shape."""
    return {
        "meta": {"vendor": "Meridian Metrics", "seed": 1,
                 "variant": "seeded", "base_date": "2026-01-05",
                 "windows": {"fit": [0, 90], "heldout": [91, 181]},
                 "synthetic": True},
        "accounts": accounts,
        "weekly_activity": weekly or [],
        "events": events or [],
        "ground_truth": {"accounts": {}, "planted_firing": {},
                         "planted_patterns": {}, "counts": {}},
    }


def account(account_id, conv_day=None):
    return {"account_id": account_id, "company": f"Co {account_id}",
            "industry": "fintech", "employees": 50,
            "converted": conv_day is not None,
            "conversion_day": conv_day,
            "conversion_date": None,
            "conversion_week": (conv_day // 7 if conv_day is not None
                                else None)}
