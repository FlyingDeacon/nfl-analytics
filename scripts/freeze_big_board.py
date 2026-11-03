"""Freeze the published big board for a season so it can be graded later.

data/derived/big_board_*.parquet is gitignored and rebuilt on demand, which is
fine for the Draft Simulator but means the board we actually published is not
recoverable after the fact. It is only reproducible while weekly.csv stops short
of the projected season; once that season's games are ingested the rebuild
answers a different question. Grading the board against what happened therefore
needs the board itself written down *before* the data to grade it with arrives.

So this writes a tracked, committed copy to data/snapshots/ as CSV — the point
is reviewability in a diff, not load speed, and 433 rows costs nothing.

An existing snapshot is never overwritten without --force. That is the whole
safety property: a frozen board that a later run can silently replace is not
frozen, and the failure would be invisible precisely because the file would
still look like a snapshot.

Run:  .venv/bin/python scripts/freeze_big_board.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
SNAPSHOTS = ROOT / "data" / "snapshots"

SEASON = 2026
SCORINGS = ("PPR", "Half PPR", "Standard")


def _slug(scoring: str) -> str:
    return scoring.replace(" ", "_")


def snapshot_path(season: int, scoring: str) -> Path:
    return SNAPSHOTS / f"big_board_{season}_preseason_{_slug(scoring)}.csv"


def freeze(season: int, scoring: str, force: bool) -> bool:
    """Rebuild one board and write it to data/snapshots/. True if written."""
    dest = snapshot_path(season, scoring)
    if dest.exists() and not force:
        print(f"skip  {dest.name}: already frozen (--force to replace)")
        return False

    # Rebuilt rather than copied off disk: the parquets are written whenever
    # someone opens the page, so their age is unknown and they can predate the
    # current model code. Two of the three were stale this way when this script
    # was added, carrying a replacement baseline the page no longer uses.
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_big_boards import build

    build(scoring)
    src = DERIVED / f"big_board_{_slug(scoring)}.parquet"
    if not src.exists():
        print(f"FAIL  {scoring}: {src.name} was not produced")
        return False

    board = pd.read_parquet(src)
    if board.empty:
        print(f"FAIL  {scoring}: board built empty, refusing to freeze")
        return False

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    board.to_csv(dest, index=False)
    print(f"froze {dest.name}: {len(board)} players")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--scoring", default=None, help="PPR | Half PPR | Standard")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing snapshot")
    args = ap.parse_args()

    targets = [args.scoring] if args.scoring else list(SCORINGS)
    for scoring in targets:
        freeze(args.season, scoring, args.force)


if __name__ == "__main__":
    main()
