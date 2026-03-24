import io
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests

# Schedules include future/current seasons; weekly stats only go through
# the last completed season that nflverse has published.
SCHEDULE_YEARS = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]
WEEKLY_YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]

WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{year}.parquet"
)


def _fetch_weekly_year(year: int) -> pd.DataFrame:
    url = WEEKLY_URL.format(year=year)
    resp = requests.get(url, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading schedules...")
    schedules = nfl.import_schedules(SCHEDULE_YEARS)

    print("Loading team descriptions...")
    teams = nfl.import_team_desc()

    print(f"schedules: {schedules.shape}  teams: {teams.shape}")
    schedules.to_csv(raw_dir / "schedules.csv", index=False)
    teams.to_csv(raw_dir / "teams.csv", index=False)

    print("Loading weekly player data (year by year)...")
    frames = []
    for yr in WEEKLY_YEARS:
        try:
            frames.append(_fetch_weekly_year(yr))
            print(f"  {yr}: ok")
        except Exception as e:
            print(f"  {yr}: skipped — {e}")

    if frames:
        weekly = pd.concat(frames, ignore_index=True)
        print(f"weekly total: {weekly.shape}")
        weekly.to_csv(raw_dir / "weekly.csv", index=False)
    else:
        print("No weekly data fetched — weekly.csv not written.")

    print(f"\nSaved to {raw_dir}")


if __name__ == "__main__":
    main()
