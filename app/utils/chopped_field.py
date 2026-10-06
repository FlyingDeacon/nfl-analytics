"""The league's own sheet, read as a standings table.

CHOPPED publishes a "mobile lookup" tab: one row per entry, one column per
week, each cell either "<Team> OK" or "<Team> X". That is enough to reconstruct
the two things the rules actually rank people on — whether you are still in,
and what you have left — so this turns it into a leaderboard rather than a grid
you have to read sideways on a phone.
"""
from __future__ import annotations

import pandas as pd

SHEET_ID = "1dy9w4VW1YIp0vfsYgRhPrlN2AKCZ2-gFhWfKeUziAng"
MOBILE_GID = "961423589"
SHEET_URL = (f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
             f"/export?format=csv&gid={MOBILE_GID}")

# The sheet writes city names; everything else in the app speaks abbreviations.
NAME_TO_ABBR = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "Kansas City": "KC", "LA Chargers": "LAC", "LA Rams": "LAR",
    "Las Vegas": "LV", "Miami": "MIA", "Minnesota": "MIN", "New England": "NE",
    "New Orleans": "NO", "NY Giants": "NYG", "NY Jets": "NYJ",
    "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Francisco": "SF",
    "Seattle": "SEA", "Tampa Bay": "TB", "Tennessee": "TEN", "Washington": "WAS",
}

WEEK_COLS = [f"W{i}" for i in range(1, 19)]
HEADER_ROW = 5          # the tab carries five rows of title and instructions


def load_field(url: str = SHEET_URL) -> pd.DataFrame:
    """Fetch the mobile-lookup tab. Raises if the sheet is unreachable."""
    raw = pd.read_csv(url, skiprows=HEADER_ROW)
    return raw.dropna(subset=["Player"])


def _parse_row(row: pd.Series) -> tuple[set[str], int, int]:
    """Teams burned, losses taken, and weeks played, from one entry's row.

    A pick is burned whether it won or lost — that is the whole difficulty of
    the format — so the used set does not care about the suffix. The suffix is
    only read to count losses, which is what the mulligan is spent on.
    """
    used: set[str] = set()
    losses = weeks = 0
    for col in WEEK_COLS:
        cell = row.get(col)
        if not isinstance(cell, str) or not cell.strip():
            continue
        name, _, mark = cell.strip().rpartition(" ")
        abbr = NAME_TO_ABBR.get(name)
        if abbr is None:
            continue
        used.add(abbr)
        weeks += 1
        if mark.upper() == "X":
            losses += 1
    return used, losses, weeks


def leaderboard(field: pd.DataFrame, team_wins: pd.Series) -> pd.DataFrame:
    """Rank the league the way the pot is actually settled.

    Three keys, in the order the rules apply them:

    1. Still alive. Nothing else matters if you are chopped.
    2. Mulligan intact. Not a tiebreaker in the rulebook, but it is the single
       biggest difference between two live entries: one of them can absorb a
       bad Sunday and the other cannot.
    3. Reserve — the average projected final wins of the teams you have NOT
       spent. This is the literal tiebreaker: the pot is never split, and when
       several entries survive it goes to the best average among unused teams.

    Reserve is worth reading carefully, because it runs the opposite way to
    intuition. Everyone alive has spent the same number of teams, and the 32
    win totals sum to a constant, so a high reserve means you have been winning
    with cheap teams. Entries that have already taken a loss tend to score
    *better* here — they burned something that lost, while the spotless entries
    got spotless by spending the best teams on the board.
    """
    rows = []
    all_teams = set(team_wins.index)
    for _, r in field.iterrows():
        used, losses, weeks = _parse_row(r)
        free = sorted(all_teams - used)
        rows.append({
            "player": r["Player"],
            "entry": r.get("Team"),
            "status": r["Status"],
            "alive": r["Status"] != "Chopped",
            "mulligan": losses == 0,
            "weeks": weeks,
            "used": len(used),
            "reserve": float(team_wins[free].mean()) if free else 0.0,
            "burned": ", ".join(sorted(used)),
        })
    out = pd.DataFrame(rows)
    return (out.sort_values(["alive", "mulligan", "reserve"],
                            ascending=[False, False, False])
               .reset_index(drop=True))
