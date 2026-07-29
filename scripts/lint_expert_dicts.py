"""Validate the hand-maintained "expert adjustment" dictionaries in the
Fantasy Predictions page against the live player-name universe.

Every dict in `app/pages/7_Fantasy_Predictions.py` (INJURY_RISK_MAP,
PLAYER_BIRTH_YEARS, PLAYER_MULTIPLIERS, EXPERT_TEAM_CORRECTIONS,
FORCE_INCLUDE_STARTERS) is keyed on an exact `player_display_name` string.
If a name format drifts (e.g. a "Jr." suffix nflverse adds/drops between
seasons) the dict entry silently stops applying — no error, no warning,
just a wrong projection. This script flags orphaned keys so that kind of
bug gets caught before it ships instead of found by manual audit.

Usage:
    python scripts/lint_expert_dicts.py
Exit code 1 if any orphaned keys are found, 0 otherwise.
"""
from __future__ import annotations

import difflib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "app" / "pages" / "7_Fantasy_Predictions.py"

# Dicts keyed by an EXACT player_display_name match (a name-format drift
# silently breaks the lookup).
EXACT_MATCH_DICTS = [
    "INJURY_RISK_MAP",
    "PLAYER_BIRTH_YEARS",
    "PLAYER_MULTIPLIERS",
]
# EXPERT_TEAM_CORRECTIONS is applied via `.str.contains(fragment)`, i.e. a
# SUBSTRING match against player_display_name — so "Kenneth Walker" legitimately
# matches "Kenneth Walker III" by design. Check containment, not equality.
SUBSTRING_MATCH_DICTS = ["EXPERT_TEAM_CORRECTIONS"]
# Keyed by team abbreviation, not player name — not checked against the
# player universe at all.
TEAM_KEYED = ["NEW_HC_PENALTY"]


def _load_module():
    """Headlessly import the Predictions page (same trick as build_big_boards.py)."""
    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(0, str(ROOT))
    import streamlit as st

    orig_radio = st.sidebar.radio
    orig_page_cfg = st.set_page_config
    st.sidebar.radio = lambda label, options, *a, **k: (
        "PPR" if "scoring" in str(label).lower() else (options[0] if len(options) else None)
    )
    st.set_page_config = lambda *a, **k: None

    try:
        spec = importlib.util.spec_from_file_location("fantasy_pred_lint", PAGE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        st.sidebar.radio = orig_radio
        st.set_page_config = orig_page_cfg


def main() -> int:
    mod = _load_module()

    name_col = mod.name_col
    board_names = set(mod.all_preds[name_col].dropna().astype(str))

    # Names that are legitimately absent from the board on purpose, or
    # sourced from a different name space than weekly.csv.
    valid_universe = (
        board_names
        | set(mod.EXPERT_REMOVE)
        | set(mod.FORCE_INCLUDE_STARTERS.keys())
    )
    rookies_csv = ROOT / "data" / "raw" / "rookies_2026.csv"
    if rookies_csv.exists():
        import pandas as pd
        valid_universe |= set(pd.read_csv(rookies_csv)["player"].dropna().astype(str))

    had_orphans = False
    for dict_name in EXACT_MATCH_DICTS + SUBSTRING_MATCH_DICTS + TEAM_KEYED:
        d = getattr(mod, dict_name, None)
        if not d:
            continue

        if dict_name in TEAM_KEYED:
            print(f"OK   {dict_name}: team-keyed, not checked against player universe")
            continue
        elif dict_name in SUBSTRING_MATCH_DICTS:
            orphans = [k for k in d if not any(k.lower() in n.lower() for n in valid_universe)]
        else:
            orphans = [k for k in d if k not in valid_universe]

        if not orphans:
            print(f"OK   {dict_name}: all {len(d)} keys match the player universe")
            continue
        had_orphans = True
        print(f"WARN {dict_name}: {len(orphans)} orphaned key(s) out of {len(d)}")
        for k in orphans:
            close = difflib.get_close_matches(k, valid_universe, n=1, cutoff=0.7)
            suggestion = f" -> did you mean {close[0]!r}?" if close else " (not on board — dead entry?)"
            print(f"       {k!r}{suggestion}")

    return 1 if had_orphans else 0


if __name__ == "__main__":
    sys.exit(main())
