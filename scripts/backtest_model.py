"""
Backtest the fantasy projection model against 2024 actual outcomes.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Load actual 2024 outcomes
print("Loading 2024 actual data...")
actual_2024 = pd.read_csv("data/raw/weekly.csv")
actual_2024 = actual_2024[
    (actual_2024["season"] == 2024) & 
    (actual_2024["season_type"] == "REG")
].copy()

# Aggregate actual points and games
actual_agg = actual_2024.groupby(["player_name", "position"]).agg({
    "fantasy_points_ppr": "sum",
    "week": "count"
}).reset_index()
actual_agg.columns = ["player_name", "position", "actual_pts", "games_played"]
actual_agg["actual_ppg"] = (actual_agg["actual_pts"] / actual_agg["games_played"]).round(2)

print(f"Loaded {len(actual_agg)} players from 2024 actual data")
print("\nSample of 2024 actuals:")
print(actual_agg.head(10))
