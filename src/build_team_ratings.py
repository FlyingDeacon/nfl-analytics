import pandas as pd
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    schedules = pd.read_csv(raw_dir / "schedules.csv")

    # Only completed games
    games = schedules[schedules["result"].notna()].copy()

    # Home team stats
    home = games[["season", "week", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["season", "week", "team", "points_for", "points_against"]

    # Away team stats
    away = games[["season", "week", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["season", "week", "team", "points_for", "points_against"]

    # Combine
    team_games = pd.concat([home, away], ignore_index=True)

    # Aggregate
    ratings = (
        team_games.groupby(["season", "team"], as_index=False)
        .agg(
            games=("week", "count"),
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
        )
    )

    # Metrics
    ratings["ppg"] = ratings["points_for"] / ratings["games"]
    ratings["oppg"] = ratings["points_against"] / ratings["games"]
    ratings["net_ppg"] = ratings["ppg"] - ratings["oppg"]

    # Sort
    ratings = ratings.sort_values("net_ppg", ascending=False)

    # Save
    ratings.to_csv(processed_dir / "team_ratings.csv", index=False)

    print("\nTop Teams:")
    print(ratings.head(10))

    print("\nSaved to data/processed/team_ratings.csv")


if __name__ == "__main__":
    main()