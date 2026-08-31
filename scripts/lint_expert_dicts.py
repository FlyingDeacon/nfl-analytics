"""Validate the hand-maintained "expert adjustment" dictionaries in the
Fantasy Predictions page against the live player-name universe.

Every dict in `app/pages/7_Fantasy_Predictions.py` (INJURY_RISK_MAP,
PLAYER_BIRTH_YEARS, PLAYER_MULTIPLIERS, EXPERT_TEAM_CORRECTIONS,
FORCE_INCLUDE_STARTERS) is keyed on an exact `player_display_name` string.
If a name format drifts (e.g. a "Jr." suffix nflverse adds/drops between
seasons) the dict entry silently stops applying — no error, no warning,
just a wrong projection. This script flags orphaned keys so that kind of
bug gets caught before it ships instead of found by manual audit.

It also checks that PLAYER_MULTIPLIERS actually *land*. A key can match a real
player and still do nothing: the multiplier is applied at step 8, but PEAK_CAP
(step 9) then clamps the rate to the player's own best season, and
REGRESS_MIN_GAMES shrinks thin samples toward the positional median. Bucky
Irving at 1.06x moved his projection by exactly zero — his pre-cap rate was
18.74 PPG against a 15.10 cap — and nothing anywhere reported it. To measure the
realized effect the board is built twice, once with the dict neutralised, and
the two per-game rates compared.

Usage:
    python scripts/lint_expert_dicts.py
Exit code 1 if any orphaned keys or swallowed multipliers are found.
"""
from __future__ import annotations

import difflib
import importlib.util
import sys
import tempfile
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


def _blank_multipliers(src: str) -> str:
    """Return `src` with the PLAYER_MULTIPLIERS literal replaced by an empty dict.

    The dict is a module-level literal that closes on a lone `}` at column 0, so
    the block runs from its opening line to the first such line after it.
    """
    lines = src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("PLAYER_MULTIPLIERS"))
    end = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
    return "".join(
        lines[:start] + ["PLAYER_MULTIPLIERS: dict[str, float] = {}\n"] + lines[end + 1:]
    )


def _load_module(name: str = "fantasy_pred_lint", source: Path | None = None,
                 write_board: bool = True):
    """Headlessly import the Predictions page (same trick as build_big_boards.py).

    `source` swaps in a rewritten copy of the page; `write_board=False` stubs out
    to_parquet so a neutralised build cannot clobber the real big board.
    """
    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    import streamlit as st

    orig_radio = st.sidebar.radio
    orig_page_cfg = st.set_page_config
    orig_parquet = pd.DataFrame.to_parquet
    st.sidebar.radio = lambda label, options, *a, **k: (
        "PPR" if "scoring" in str(label).lower() else (options[0] if len(options) else None)
    )
    st.set_page_config = lambda *a, **k: None
    if not write_board:
        pd.DataFrame.to_parquet = lambda self, *a, **k: None

    try:
        spec = importlib.util.spec_from_file_location(name, str(source or PAGE))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        st.sidebar.radio = orig_radio
        st.set_page_config = orig_page_cfg
        pd.DataFrame.to_parquet = orig_parquet


def _check_multiplier_effect(mod) -> int:
    """Report PLAYER_MULTIPLIERS entries whose realized effect != the value set.

    Returns the number of overrides the pipeline swallowed.
    """
    mults = {k: v for k, v in mod.PLAYER_MULTIPLIERS.items() if v != 1.0}
    if not mults:
        print("OK   PLAYER_MULTIPLIERS: nothing to measure")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        neutral = Path(tmp) / "fp_neutral.py"
        neutral.write_text(_blank_multipliers(PAGE.read_text()))
        base_mod = _load_module("fantasy_pred_neutral", neutral, write_board=False)

    name_col = mod.name_col
    after = mod.all_preds.drop_duplicates(name_col).set_index(name_col)["pred_ppg"]
    before = base_mod.all_preds.drop_duplicates(name_col).set_index(name_col)["pred_ppg"]

    swallowed = []
    for name, intended in mults.items():
        if name not in after.index or name not in before.index:
            continue                      # orphan check above already covers this
        b, a = float(before[name]), float(after[name])
        if b <= 0:
            continue
        realized = a / b
        # Share of the intended move that survived the cap and the shrinkage.
        # 1.0 = applied in full, 0.0 = the override did literally nothing.
        landed = (realized - 1.0) / (intended - 1.0)
        if abs(realized - intended) > 0.02 and landed < 0.75:
            swallowed.append((landed, name, intended, realized, b, a))

    if not swallowed:
        print(f"OK   PLAYER_MULTIPLIERS: all {len(mults)} overrides land as written")
        return 0

    swallowed.sort()
    print(f"WARN PLAYER_MULTIPLIERS: {len(swallowed)} of {len(mults)} override(s) "
          f"only partly survive PEAK_CAP / regression (worst first)")
    for landed, name, intended, realized, b, a in swallowed:
        print(f"       {name!r}: set {intended:.2f}x, got {realized:.2f}x — "
              f"{landed * 100:3.0f}% of the move landed ({b:.2f} -> {a:.2f} PPG)")
    print("       fix: use PROJ_GAMES_OVERRIDES, or add to PEAK_CAP_EXEMPT, or "
          "recalibrate the value against the realized effect")
    return len(swallowed)


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

    swallowed = _check_multiplier_effect(mod)
    return 1 if (had_orphans or swallowed) else 0


if __name__ == "__main__":
    sys.exit(main())
