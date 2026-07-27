"""Fetch / normalize NFL season win totals into data/raw/win_totals_<year>.csv.

Real Vegas win totals live on sportsbook + media pages that are JavaScript-
rendered (so they don't scrape cleanly) or behind The Odds API's paid futures
tier.  There is no reliable free CSV of season win totals, so this tool gives two
paths:

  1. Odds API  (best-effort, needs a key)
       ODDS_API_KEY=xxxx python scripts/fetch_win_totals.py --source odds-api
     Pulls the NFL season-wins market if your plan exposes it; exits with a clear
     message if it doesn't.

  2. Normalize a raw list  (always works — the reliable path)
       python scripts/fetch_win_totals.py --source path/to/raw.txt
     `raw` is any "Team <number>" list (one per line, full names or abbrevs, the
     ": " / tab / comma separators all work) — e.g. paste the 32 lines straight
     from FOX/OddsShark/BetMGM.  It maps team names to our abbreviations,
     validates that all 32 are present, and writes the CSV in canonical form.

Output columns: team,win_total  (matching data/raw/win_totals_2026.csv).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Map any common team spelling / city / nickname / alias -> our abbreviation.
NAME_TO_ABBR = {
    "cardinals": "ARI", "arizona": "ARI", "ari": "ARI",
    "falcons": "ATL", "atlanta": "ATL", "atl": "ATL",
    "ravens": "BAL", "baltimore": "BAL", "bal": "BAL",
    "bills": "BUF", "buffalo": "BUF", "buf": "BUF",
    "panthers": "CAR", "carolina": "CAR", "car": "CAR",
    "bears": "CHI", "chicago": "CHI", "chi": "CHI",
    "bengals": "CIN", "cincinnati": "CIN", "cin": "CIN",
    "browns": "CLE", "cleveland": "CLE", "cle": "CLE",
    "cowboys": "DAL", "dallas": "DAL", "dal": "DAL",
    "broncos": "DEN", "denver": "DEN", "den": "DEN",
    "lions": "DET", "detroit": "DET", "det": "DET",
    "packers": "GB", "green bay": "GB", "gb": "GB", "gnb": "GB",
    "texans": "HOU", "houston": "HOU", "hou": "HOU",
    "colts": "IND", "indianapolis": "IND", "ind": "IND",
    "jaguars": "JAX", "jacksonville": "JAX", "jax": "JAX", "jac": "JAX",
    "chiefs": "KC", "kansas city": "KC", "kc": "KC", "kan": "KC",
    "raiders": "LV", "las vegas": "LV", "lv": "LV", "oakland": "LV", "oak": "LV",
    "chargers": "LAC", "la chargers": "LAC", "los angeles chargers": "LAC",
    "lac": "LAC", "san diego": "LAC",
    "rams": "LAR", "la rams": "LAR", "los angeles rams": "LAR", "lar": "LAR",
    "la": "LAR", "st. louis": "LAR", "st louis": "LAR",
    "dolphins": "MIA", "miami": "MIA", "mia": "MIA",
    "vikings": "MIN", "minnesota": "MIN", "min": "MIN",
    "patriots": "NE", "new england": "NE", "ne": "NE", "nwe": "NE",
    "saints": "NO", "new orleans": "NO", "no": "NO", "nor": "NO",
    "giants": "NYG", "ny giants": "NYG", "new york giants": "NYG", "nyg": "NYG",
    "jets": "NYJ", "ny jets": "NYJ", "new york jets": "NYJ", "nyj": "NYJ",
    "eagles": "PHI", "philadelphia": "PHI", "phi": "PHI",
    "steelers": "PIT", "pittsburgh": "PIT", "pit": "PIT",
    "49ers": "SF", "san francisco": "SF", "sf": "SF", "sfo": "SF", "niners": "SF",
    "seahawks": "SEA", "seattle": "SEA", "sea": "SEA",
    "buccaneers": "TB", "bucs": "TB", "tampa bay": "TB", "tampa": "TB", "tb": "TB",
    "titans": "TEN", "tennessee": "TEN", "ten": "TEN",
    "commanders": "WAS", "washington": "WAS", "was": "WAS", "wsh": "WAS",
}
ALL_ABBRS = sorted(set(NAME_TO_ABBR.values()))


def _match_abbr(text: str) -> str | None:
    t = text.lower().strip()
    if t in NAME_TO_ABBR:
        return NAME_TO_ABBR[t]
    # Longest-key substring match (so "Los Angeles Rams" beats "la").
    for key in sorted(NAME_TO_ABBR, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", t):
            return NAME_TO_ABBR[key]
    return None


def _parse_raw(raw: str) -> dict[str, float]:
    """Parse arbitrary 'Team <number>' lines into {abbr: win_total}."""
    totals: dict[str, float] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", line.replace(",", " ").rstrip("."))
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)", line)
        if not m:
            continue
        num = float(m.group(1))
        name = line[: line.rfind(m.group(1))]
        name = re.sub(r"^\s*\d+[.)]\s*", "", name)  # strip leading list index
        abbr = _match_abbr(name)
        if abbr and 0 <= num <= 17:
            totals[abbr] = num
    return totals


def _write(totals: dict[str, float], out: Path) -> None:
    missing = [a for a in ALL_ABBRS if a not in totals]
    if missing:
        print(f"ERROR: missing {len(missing)} team(s): {', '.join(missing)}",
              file=sys.stderr)
        print("Refusing to write a partial file.", file=sys.stderr)
        sys.exit(1)
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = ["team,win_total"] + [
        f"{a},{v:g}" for a, v in ordered
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(ordered)} teams -> {out}")


def _from_odds_api(year: int) -> dict[str, float]:
    import json
    import urllib.request

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ERROR: set ODDS_API_KEY to use --source odds-api", file=sys.stderr)
        sys.exit(1)
    base = "https://api.the-odds-api.com/v4/sports"
    with urllib.request.urlopen(f"{base}/?apiKey={key}") as r:
        sports = json.load(r)
    keys = [s["key"] for s in sports if "americanfootball_nfl" in s["key"]]
    win_key = next((k for k in keys if "win" in k.lower() or "total" in k.lower()), None)
    if not win_key:
        print("Season win totals are not exposed on your Odds API plan "
              f"(NFL keys seen: {keys}).", file=sys.stderr)
        print("Use the normalize path instead: pass a raw list file as --source.",
              file=sys.stderr)
        sys.exit(2)
    url = f"{base}/{win_key}/odds/?apiKey={key}&regions=us&markets=totals&oddsFormat=american"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    totals: dict[str, float] = {}
    for ev in data:
        abbr = _match_abbr(ev.get("home_team", "") or ev.get("description", ""))
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    pt = oc.get("point")
                    a = abbr or _match_abbr(oc.get("description", "") or oc.get("name", ""))
                    if a is not None and pt is not None:
                        totals[a] = float(pt)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="'odds-api' or a path to a raw 'Team <number>' list")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    out = REPO_ROOT / "data" / "raw" / f"win_totals_{args.year}.csv"
    if args.source == "odds-api":
        totals = _from_odds_api(args.year)
    else:
        raw = Path(args.source).read_text()
        totals = _parse_raw(raw)
    _write(totals, out)


if __name__ == "__main__":
    main()
