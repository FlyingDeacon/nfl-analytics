"""Headless builder for the Fantasy big board(s).

The Draft Simulator consumes a parquet snapshot of the exact big board the
Fantasy Predictions page produces. That page normally writes the parquet as a
side effect when a user opens it in Streamlit. This script reproduces the same
build *without* a browser by importing the page module in Streamlit "bare mode"
(widget calls return defaults, which we override via monkeypatch) so the board
can be generated on demand — e.g. the first time someone opens the simulator.

Usage:
    python scripts/build_big_boards.py --scoring PPR
    python scripts/build_big_boards.py --all
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "app" / "pages" / "7_Fantasy_Predictions.py"
SCORINGS = ("PPR", "Half PPR", "Standard")


def build(scoring: str) -> None:
    """Execute the Predictions page module headlessly for one scoring format.

    The page reads its scoring from a sidebar radio; in bare mode that returns
    the first option, so we monkeypatch the widget functions to force the
    format we want and silence set_page_config's single-call guard.
    """
    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(0, str(ROOT))
    import streamlit as st

    orig_radio = st.sidebar.radio
    orig_page_cfg = st.set_page_config

    def fake_radio(label, options, *a, **k):
        if "scoring" in str(label).lower():
            return scoring
        return options[0] if len(options) else None

    st.sidebar.radio = fake_radio
    st.set_page_config = lambda *a, **k: None

    try:
        spec = importlib.util.spec_from_file_location(
            f"fantasy_pred_build_{scoring.replace(' ', '_')}", PAGE
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        st.sidebar.radio = orig_radio
        st.set_page_config = orig_page_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", default="PPR", help="PPR | Half PPR | Standard")
    ap.add_argument("--all", action="store_true", help="build every scoring format")
    args = ap.parse_args()

    targets = SCORINGS if args.all else [args.scoring]
    for s in targets:
        build(s)
        print(f"built big board: {s}")


if __name__ == "__main__":
    main()
