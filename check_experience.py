import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import pandas as pd
from utils.data_loader import load_weekly

weekly = load_weekly()
if weekly.empty:
    print('No data')
    exit()

# Check 2023 rookies who played in both 2024 and 2025 (3rd-year for 2026)
rookies_2023 = set(weekly[weekly['season'] == 2023]['player_display_name'].unique())
played_2024 = set(weekly[weekly['season'] == 2024]['player_display_name'].unique())
played_2025 = set(weekly[weekly['season'] == 2025]['player_display_name'].unique())

third_year_2026 = rookies_2023 & played_2024 & played_2025

print('2023 Rookies who played in 2024 AND 2025 (3rd-year for 2026):')
for player in sorted(third_year_2026):
    pos = weekly[weekly['player_display_name'] == player]['position'].iloc[0]
    print(f'  {player} ({pos})')

# Check EXPERT_MULTIPLIERS
try:
    from app.pages.Fantasy_Predictions import EXPERT_MULTIPLIERS
    multiplier_players = set(EXPERT_MULTIPLIERS.keys())
    overlaps = third_year_2026 & multiplier_players
    if overlaps:
        print('\nThese 3rd-year players have multipliers:')
        for player in sorted(overlaps):
            pos = weekly[weekly['player_display_name'] == player]['position'].iloc[0]
            mult = EXPERT_MULTIPLIERS[player]
            print(f'  {player} ({pos}): {mult}')
    else:
        print('\nNo 3rd-year players have multipliers.')
except ImportError:
    print('\nCould not import EXPERT_MULTIPLIERS')