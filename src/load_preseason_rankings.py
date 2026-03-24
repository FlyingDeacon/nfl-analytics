"""
Fetch FantasyPros PPR preseason ADP rankings for each season and save to
data/raw/preseason_rankings.csv.

Run once (or annually before a new season):
    python src/load_preseason_rankings.py
"""

import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH  = DATA_DIR / "preseason_rankings.csv"

BASE_URL = "https://www.fantasypros.com/nfl/adp/ppr-overall.php"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def _normalize_name(name: str) -> str:
    """Lowercase, remove punctuation and common suffixes for fuzzy joining."""
    name = str(name).lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = re.sub(r"[.\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _parse_player_cell(raw: str) -> tuple[str, str, str]:
    """
    Handle multiple FantasyPros cell formats:
      'Christian McCaffrey SF RB'          (old multiline collapsed)
      'Christian McCaffrey\nSF RB'         (newline separated)
      "Ja'Marr Chase CIN (10)"             (name + team abbr + bye week)
      "Malik Nabers NYG (14) O"            (+ position letter after bye)
    Returns (clean_name, team_abbr, position).
    """
    raw = str(raw).strip()

    # Format: "Player Name TEAM (bye) [POS]"
    m = re.match(r"^(.+?)\s+([A-Z]{2,4})\s*\(\d+\)\s*([A-Z]*)?\s*$", raw)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3) or ""

    # Format: multiline or double-space separated
    parts = re.split(r"\n|\s{2,}", raw)
    name = parts[0].strip()
    team, pos = "", ""
    if len(parts) > 1:
        tokens = parts[1].strip().split()
        if len(tokens) >= 2:
            team, pos = tokens[0], tokens[1]
        elif len(tokens) == 1:
            pos = tokens[0]
    return name, team, pos


def fetch_year(year: int):
    url = f"{BASE_URL}?year={year}" if year else BASE_URL
    print(f"  Fetching {year} … {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ✗ Request failed for {year}: {e}")
        return None

    try:
        tables = pd.read_html(StringIO(resp.text))
    except Exception as e:
        print(f"  ✗ Could not parse HTML for {year}: {e}")
        return None

    if not tables:
        print(f"  ✗ No tables found for {year}")
        return None

    df = tables[0]
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Identify the player / rank columns by common names
    player_col = next(
        (c for c in df.columns if "player" in c or "name" in c), None
    )
    rank_col = next(
        (c for c in df.columns if c in ("#", "rank", "overall")), None
    )
    avg_col = next(
        (c for c in df.columns if "avg" in c), None
    )

    if player_col is None:
        print(f"  ✗ Could not find player column for {year}. Columns: {list(df.columns)}")
        return None

    rows = []
    for i, row in df.iterrows():
        raw_player = str(row[player_col])
        if raw_player.lower() in ("nan", "player", ""):
            continue
        name, team, pos = _parse_player_cell(raw_player)
        if not name:
            continue

        # Prefer explicit rank column; fall back to row position
        if rank_col and pd.notna(row[rank_col]):
            try:
                rank = int(float(str(row[rank_col]).replace(".", "")))
            except ValueError:
                rank = i + 1
        elif avg_col and pd.notna(row[avg_col]):
            try:
                rank = round(float(row[avg_col]))
            except ValueError:
                rank = i + 1
        else:
            rank = i + 1

        rows.append({
            "season":          year,
            "preseason_rank":  rank,
            "player_name":     name,
            "team":            team,
            "position":        pos,
            "name_key":        _normalize_name(name),
        })

    if not rows:
        print(f"  ✗ Parsed 0 players for {year}")
        return None

    out = pd.DataFrame(rows).sort_values("preseason_rank").reset_index(drop=True)
    print(f"  ✓ {year}: {len(out)} players")
    return out


def main():
    all_frames = []
    for year in YEARS:
        df = fetch_year(year)
        if df is not None:
            all_frames.append(df)
        time.sleep(1.5)   # be polite to FantasyPros

    if not all_frames:
        print("No data fetched — nothing saved.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
