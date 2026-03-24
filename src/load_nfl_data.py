from pathlib import Path
import nfl_data_py as nfl


def main():
    years = [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading schedules...")
    schedules = nfl.import_schedules(years)

    print("Loading team descriptions...")
    teams = nfl.import_team_desc()

    print("\nShapes:")
    print("schedules:", schedules.shape)
    print("teams:", teams.shape)

    schedules.to_csv(raw_dir / "schedules.csv", index=False)
    teams.to_csv(raw_dir / "teams.csv", index=False)

    print("Loading weekly player data...")
    weekly = nfl.import_weekly_data(years)
    print("weekly:", weekly.shape)
    weekly.to_csv(raw_dir / "weekly.csv", index=False)

    print(f"\nSaved files to {raw_dir}")


if __name__ == "__main__":
    main()