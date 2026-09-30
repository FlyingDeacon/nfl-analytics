"""Pull the weekend's results into the app without dropping to a terminal.

The in-season weekly update is only two of the steps in
`scripts/update_2026_data.py`: refetch the 2026 schedule, which is where
nflverse publishes final scores, and rebuild the team ratings that are derived
from it. The third step in that script — depth charts — is offseason work that
moves on roster news rather than on Sunday, and it is the slow half of the
download, so it is deliberately not run here.

Both underlying steps are imported from their existing homes rather than
reimplemented. They are idempotent (the 2026 rows are dropped and rewritten
wholesale), so pressing the button twice is harmless.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SEASON = 2026


def _load(path: Path, name: str):
    """Import a module by file path.

    `scripts/` and `src/` are plain directories rather than packages, so they
    cannot be imported normally from inside `app/`.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def results_status(schedule: pd.DataFrame, week: int) -> dict:
    """How much of `week` is in the books, per the schedule on disk.

    Drives the button's caption so it can say "Week 3 is final" rather than
    leaving you to guess whether pressing it will achieve anything.
    """
    wk = schedule[(schedule["season"] == SEASON)
                  & (schedule["game_type"] == "REG")
                  & (schedule["week"] == week)]
    played = int(wk["result"].notna().sum())
    return {"week": week, "played": played, "total": int(len(wk)),
            "complete": bool(len(wk)) and played == len(wk)}


def pull_results() -> str:
    """Refetch 2026 scores and rebuild team ratings. Returns a one-line summary.

    Raises whatever the download raises — the caller surfaces it rather than
    swallowing it, because a silent no-op that looks like success is the one
    outcome worse than an error here.
    """
    sched_mod = _load(ROOT / "scripts" / "update_2026_data.py", "_upd2026")
    ratings_mod = _load(ROOT / "src" / "build_team_ratings.py", "_buildratings")

    sched_mod.update_schedule()
    ratings_mod.main()

    sched = pd.read_csv(ROOT / "data" / "raw" / "schedules.csv", low_memory=False)
    reg = sched[(sched["season"] == SEASON) & (sched["game_type"] == "REG")]
    done = reg[reg["result"].notna()]
    last = int(done["week"].max()) if len(done) else 0
    return (f"Pulled {len(done)} final results through Week {last}, "
            f"and rebuilt team ratings.")
