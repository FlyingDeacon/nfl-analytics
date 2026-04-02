import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Fantasy Predictions · NFL", page_icon="🔮", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

render_sidebar_nav(current_page="7_Fantasy_Predictions")

if st.button("← Back to Fantasy Football", key="back_btn"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔮</div>
    <div>
        <div class="title">2026 Fantasy Predictions</div>
        <div class="subtitle">Per-game regression · injury-adjusted playing time · recency-weighted training</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TARGET_COL = "fantasy_points_ppr"
PREDICTION_YEAR = 2026
NFL_GAMES = 17
DEFAULT_PROJ_GAMES = 14.0   # fallback when prior-season data is absent

# QBs need 12 full games to qualify as a genuine starter (not a backup fill-in).
# Skill positions use 6 — role players and injury-return candidates are valid.
MIN_GAMES_BY_POS = {"QB": 12, "RB": 6, "WR": 6, "TE": 6}
MAX_PROJ_GAMES   = 16       # conservative ceiling (no one is guaranteed 17)

# ── Value Over Replacement (VOR) — positional scarcity scoring ───────────────
# Replacement level = projected points of the last startable player at each position
# in a 10-team PPR league, calibrated from 2025 championship team analysis.
# 10-team leagues have steeper value cliffs: elite players more scarce, waiver wire thinner.
REPLACEMENT_LEVEL = {
    "QB":  240,   # 10 starters; QB punt strategy viable
    "RB":  115,   # ~20 starters (2 per team); steep drop after elite tier
    "WR":   95,   # ~30 starters (3 per team); sharp cliff between tiers
    "TE":   80,   # 10 starters; elite TE commands significant premium
}

# ── 2026 CONSENSUS ADP (10-team PPR, pre-draft) ──────────────────────────────
# Overall pick numbers (1 = first overall).
# Source: FantasyPros / Sleeper community consensus, April 2026.
# Update as draft season approaches for live calibration.
# Players not found default to ADP = 999 (undrafted / unranked).
ADP_2026: dict[str, float] = {
    # ── Round 1 (picks 1–10) ──────────────────────────────────────────────────
    "Saquon Barkley":         1.0,
    "Justin Jefferson":       2.0,
    "CeeDee Lamb":            3.0,
    "Christian McCaffrey":    4.0,
    "Ja'Marr Chase":          5.0,
    "Bijan Robinson":         6.0,
    "Lamar Jackson":          7.0,
    "Breece Hall":            8.0,
    "De'Von Achane":          9.0,
    "Jahmyr Gibbs":          10.0,
    # ── Round 2 (picks 11–20) ─────────────────────────────────────────────────
    "Jalen Hurts":           11.0,
    "Josh Allen":            12.0,
    "Amon-Ra St. Brown":     13.0,
    "Jonathan Taylor":       14.0,
    "Derrick Henry":         15.0,
    "A.J. Brown":            16.0,
    "James Cook":            17.0,
    "Kyren Williams":        18.0,
    "Puka Nacua":            19.0,
    "Kenneth Walker III":    20.0,
    # ── Round 3 (picks 21–30) ─────────────────────────────────────────────────
    "Patrick Mahomes":       21.0,
    "Trey McBride":          22.0,
    "DK Metcalf":            23.0,
    "Tee Higgins":           24.0,
    "Josh Jacobs":           25.0,
    "Jaxon Smith-Njigba":    26.0,
    "DJ Moore":              27.0,
    "George Pickens":        28.0,
    "D'Andre Swift":         29.0,
    "Rhamondre Stevenson":   30.0,
    # ── Round 4 (picks 31–40) ─────────────────────────────────────────────────
    "C.J. Stroud":           31.0,
    "Nico Collins":          32.0,
    "Brock Bowers":          33.0,
    "Sam LaPorta":           34.0,
    "Brian Thomas":          35.0,
    "Justin Herbert":        36.0,
    "Travis Etienne":        37.0,
    "Chuba Hubbard":         38.0,
    "Terry McLaurin":        39.0,
    "Jaylen Waddle":         40.0,
    # ── Round 5 (picks 41–50) ─────────────────────────────────────────────────
    "Rico Dowdle":           41.0,
    "Mark Andrews":          42.0,
    "Travis Kelce":          43.0,
    "DeVonta Smith":         44.0,
    "Sam Darnold":           45.0,
    "Drake London":          46.0,
    "Jordan Love":           47.0,
    "Garrett Wilson":        48.0,
    "Michael Pittman":       49.0,
    "Stefon Diggs":          50.0,
    # ── Round 6 (picks 51–60) ─────────────────────────────────────────────────
    "Rashee Rice":           51.0,
    "Javonte Williams":      52.0,
    "Brian Robinson":        53.0,
    "Chris Godwin":          54.0,
    "David Njoku":           55.0,
    "Chase Brown":           56.0,
    "Dak Prescott":          57.0,
    "Kyler Murray":          58.0,
    "Courtland Sutton":      59.0,
    "Caleb Williams":        60.0,
    # ── Round 7 (picks 61–70) ─────────────────────────────────────────────────
    "Jayden Daniels":        61.0,
    "Tank Bigsby":           62.0,
    "Deebo Samuel Sr.":      63.0,
    "Dallas Goedert":        64.0,
    "Cooper Kupp":           65.0,
    "Kyle Pitts":            66.0,
    "Calvin Ridley":         67.0,
    "George Kittle":         68.0,
    "T.J. Hockenson":        69.0,
    "Isiah Pacheco":         70.0,
    # ── Round 8 (picks 71–80) ─────────────────────────────────────────────────
    "Zay Flowers":           71.0,
    "Drake Maye":            72.0,
    "Jake Ferguson":         73.0,
    "Zack Moss":             74.0,
    "Wan'Dale Robinson":     75.0,
    "Bo Nix":                76.0,
    "Justin Fields":         77.0,
    "Aaron Jones":           78.0,
    "Chris Olave":           79.0,
    "Emeka Egbuka":          80.0,
    # ── Round 9 (picks 81–90) ─────────────────────────────────────────────────
    "Pat Freiermuth":        81.0,
    "Marvin Harrison":       82.0,
    "Cole Kmet":             83.0,
    "Evan Engram":           84.0,
    "Jaylen Warren":         85.0,
    "Will Levis":            86.0,
    "Cade Otton":            87.0,
    "Jameson Williams":      88.0,
    "Khalil Shakir":         89.0,
    "Tony Pollard":          90.0,
    # ── Round 10 (picks 91–100) ───────────────────────────────────────────────
    "Isaiah Likely":         91.0,
    "Tucker Kraft":          92.0,
    "Bryce Young":           93.0,
    "Harold Fannin Jr.":     94.0,
    "Matthew Stafford":      95.0,
    "Tyler Warren":          96.0,
    "Chigoziem Okonkwo":     97.0,
    "Dameon Pierce":         98.0,
    "Michael Mayer":         99.0,
    "Jaxson Dart":          100.0,
    # ── Round 11+ (picks 101–130) ────────────────────────────────────────────
    "Joe Mixon":            101.0,
    "RJ Harvey":            102.0,
    "Ashton Jeanty":        103.0,
    "TreVeyon Henderson":   104.0,
    "DeAndre Hopkins":      105.0,
    "Keenan Allen":         106.0,
    "Cam Ward":             107.0,
    "Tyler Lockett":        108.0,
    "Kenneth Gainwell":     109.0,
    "Aaron Rodgers":        110.0,
    "Malik Willis":         111.0,
    "Diontae Johnson":      112.0,
    "Tyler Shough":         113.0,
    "David Montgomery":     114.0,
    "Kareem Hunt":          115.0,
    "Nick Chubb":           116.0,
    "Geno Smith":           117.0,
    "Daniel Jones":         118.0,
    "Jacoby Brissett":      119.0,
    "Baker Mayfield":       120.0,
    "Sam Darnold":          121.0,
    "Cam Ward":             122.0,
}

def get_adp(player_name: str) -> float:
    """Look up 2026 consensus ADP for a player. Returns 999.0 if not found.

    Tries exact match first, then case-insensitive partial match so minor
    name variations (Jr., Sr., suffixes) still resolve correctly.
    """
    if player_name in ADP_2026:
        return ADP_2026[player_name]
    # Partial / case-insensitive fallback
    name_lower = str(player_name).lower()
    for key, val in ADP_2026.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return val
    return 999.0   # undrafted / unranked

# Ridge penalty prevents wild extrapolation from small samples.
# 4.0 balances regularisation vs. tracking elite consistent performers:
# alpha=8 over-shrinks stars like Allen (15% below true level) while barely
# affecting mediocre QBs — the asymmetry creates systematic ranking errors.
RIDGE_ALPHA = 4.0
# Per-year recency decay: the 2024→2025 pair is weighted ~35 % higher than 2023→2024.
DECAY = 0.35
# Prior-year PPG blend weight. Published research shows prior-year fantasy PPG
# explains ~45-50% of the variance in the following year — making it the single
# strongest predictor. The ridge model captures multi-year trends and component
# stats (the other ~55%). Blending both gives a 2-model ensemble that mirrors
# how professional projection systems (4for4, PFF, FantasyPros) are built.
PPG_BLEND_WEIGHT = 0.45  # 45% prior-year PPG, 55% ridge model

# Backtest-calibrated expected-games used ONLY for the PPG baseline component.
# Projecting 17 games over-estimates because not all starters play a full season.
# Backtesting 2024→2025 (train 2016-2023 only) found these scales minimise bias:
#   QB: PPG×15 → bias ≈ −2 pts  (vs +44 at 17 games)  — QBs average 13-15 starts
#   RB: PPG×15 → bias ≈ +3 pts  (vs +10 at 17 games)  — RBs high injury/rotation rate
#   WR: PPG×15 → bias reduced   (role-changers & injuries common at 17+)
#   TE: PPG×16 → slight reduction (TEs more durable than RB/WR)
# NOTE: proj_games (17) is still used for the model component and the "Proj GP" display.
# This dict only reduces the PPG anchor; it doesn't cap the final projected total.
PPG_BASELINE_GAMES: dict[str, int] = {
    "QB": 15,
    "RB": 15,
    "WR": 15,
    "TE": 16,
}

# Per-game features only — normalises out "played more games = more counting stats".
# game_rate (games/17) is always appended and captures injury-proneness / role depth.
POSITION_FEATURES = {
    "QB": ["passing_yards", "passing_tds", "interceptions",
           "rushing_yards", "rushing_tds", "completions", "attempts"],
    "RB": ["rushing_yards", "rushing_tds", "carries",
           "receptions", "receiving_yards", "receiving_tds", "targets"],
    "WR": ["receiving_yards", "receiving_tds", "targets", "receptions", "rushing_yards"],
    "TE": ["receiving_yards", "receiving_tds", "targets", "receptions"],
}
POSITION_LABELS = {"QB": "Quarterbacks", "RB": "Running Backs",
                   "WR": "Wide Receivers",  "TE": "Tight Ends"}

# ══════════════════════════════════════════════════════════════════════════════
# NFL EXPERT ADJUSTMENTS — 2026 roster intelligence
# Applied as post-model corrections on top of the statistical projection.
# ══════════════════════════════════════════════════════════════════════════════

# Confirmed 2026 starters who fall below the model's games minimum.
# Format: player_name → (player_id, position, 2026_team, manual_ppg_or_None)
#   manual_ppg: use when ALL historical seasons are backup-level (no qualifying rate exists).
#               Set to None to let the model find the last qualifying season automatically.
FORCE_INCLUDE_STARTERS = {
    "Kyler Murray":     ("00-0035228", "QB", "MIN", None),   # 5 games 2025 (ARI injury); uses 2024 full season
    "Malik Willis":     ("00-0038128", "QB", "MIA", 15.5),   # career backup turned starter — expert PPG
    "Tyler Shough":     ("00-0040743", "QB", "NO",  16.0),   # NO QB1; 10 games 2025 (below 12 QB min); ~16 PPG
    "Matthew Stafford": ("00-0026498", "QB", "LAR", None),   # Returning for 2026 with LAR; find most recent qualifying season
    "Jayden Daniels":   ("00-0039910", "QB", "WAS", None),   # 7 games 2025 (injury); uses 2024 full season (20.93 PPG)
}

# Players removed from 2026 board (not projected starters / retired / injury risk)
EXPERT_REMOVE = {
    "Kirk Cousins",      # Not a projected 2026 starter
    "Rob Gronkowski",    # Officially retired March 2026
    "Michael Penix",     # ACL surgery (Nov 2025); intended ATL starter (~60% Week 1) but removed pending recovery clearance
    "Tua Tagovailoa",    # ATL backup/placeholder — NOT a 2026 starter; Penix is intended starter
    "Alvin Kamara",      # Demoted to NO RB2 behind Travis Etienne; also only 11 games in 2025 (below 6-game floor)
    "Austin Ekeler",     # Torn Achilles; out for 2026 season
    "Malik Nabers",      # ACL in 2025; only 4 games played — below 6-game WR minimum
}

# Injury risk mapping — 2026 season outlook (HIGH/MEDIUM/LOW risk levels)
# Based on NFL Expert research as of March 2026. Applied to RB/WR/TE players.
# QBs use historical games-played average; skill positions use this expert lookup.
# Values use same padding as QB injury_risk display for consistency.
INJURY_RISK_MAP = {
    # HIGH RISK — expected to miss significant 2026 time or have uncertain availability
    "Brandon Aiyuk":      "      Yes      ",   # ACL/MCL tear; missed entire 2025 season
    "Tyreek Hill":        "      Yes      ",   # Multi-ligament knee injury (ACL+); franchise released him
    "Tank Dell":          "      Yes      ",   # Multi-ligament knee injury (ACL/MCL/LCL/meniscus); missed 2025
    "Christian Watson":   "      Yes      ",   # ACL tear (Jan 2025); expected to miss significant 2026 time

    # MEDIUM RISK — may miss games early season or have recurring issues
    "Stefon Diggs":       "      Yes      ",   # ACL recovery (2024); age + team transition risk
    "George Kittle":      "      Yes      ",   # Achilles injury (Jan 2026); age 33; high tear but positive recovery
    "Sam LaPorta":        "      Yes      ",   # Back surgery (Nov 2024); recurring back injury risk
    "Brock Bowers":       "      Yes      ",   # PCL injury (Week 1 2025); may miss early 2026
    "Dalton Kincaid":     "      Yes      ",   # Recurring PCL issues; missed 5 games in 2025
}

# Team corrections: player name fragment → corrected 2026 team abbreviation
# Sources: ESPN / NFL.com free agency trackers, March 2026
EXPERT_TEAM_CORRECTIONS = {
    "Travis Etienne":    "NO",   # Signed with New Orleans Saints (left JAX)
    "Kyler Murray":      "MIN",  # 1-year deal with Vikings
    "Jaylen Waddle":     "DEN",  # Traded MIA → DEN (pairs with Bo Nix)
    "Michael Pittman":   "PIT",  # Traded IND → PIT
    "DJ Moore":          "BUF",  # Traded CHI → BUF (Josh Allen boost)
    "Malik Willis":      "MIA",  # 3-yr $67.5M / $45M guaranteed — CONFIRMED MIA starter
    "Kenneth Walker":    "KC",   # 3-yr $43M deal — joins Kansas City
    "Mike Evans":        "SF",   # 3-yr $60M deal — joins 49ers
    "Derrick Henry":     "BAL",  # Re-signed with Baltimore Ravens
    "Sam Darnold":       "SEA",  # Signed with Seattle Seahawks
    "Keenan Allen":      "LAC",  # Returns to the Chargers
    "DeAndre Hopkins":   "BAL",  # Signed with Baltimore Ravens; pairs with Lamar
    "Rico Dowdle":       "PIT",  # Signed with Pittsburgh Steelers (was DAL)
    "George Pickens":    "DAL",  # Traded PIT → DAL; pairs with Dak Prescott
    "Tyler Shough":      "NO",   # Confirmed New Orleans Saints starter 2026
}

# ── NEW HEAD COACH PENALTY ───────────────────────────────────────────────────
# 10 teams changed head coaches for the 2026 season (source: NFL.com / Yahoo Sports).
# New systems create uncertainty for all skill-position players regardless of talent.
# Two-tier penalty based on head-coaching experience level:
#
#  -4% (0.96) — Proven HC with prior NFL head-coaching success; lower disruption risk
#    NYG  John Harbaugh  (BAL 2008-2025; 1 Super Bowl)
#    ATL  Kevin Stefanski (CLE 2020-2024; 2× NFL Coach of the Year)
#    PIT  Mike McCarthy  (GB 2006-2018 + DAL 2020-2024; 1 Super Bowl)
#    TEN  Frank Reich    (IND 2018-2022; playoff experience)
#
#  -7% (0.93) — First-time or limited HC experience; highest system uncertainty
#    BUF  Joe Brady      (promoted from OC; first HC role)
#    BAL  Jesse Minter   (LAC DC → first HC role)
#    MIA  Jeff Hafley    (college HC → first NFL HC role)
#    CLE  Todd Monken    (Ravens OC → first HC role)
#    LV   Klint Kubiak   (NO/SEA OC → first HC role)
#    ARI  Mike LaFleur   (LAR OC → first HC role)

NEW_HC_PENALTY = {
    # Experienced — lower uncertainty
    "NYG": 0.96,   # John Harbaugh
    "ATL": 0.96,   # Kevin Stefanski
    "PIT": 0.96,   # Mike McCarthy
    "TEN": 0.96,   # Frank Reich
    # First-time / limited — higher uncertainty
    "BUF": 0.93,   # Joe Brady
    "BAL": 0.93,   # Jesse Minter
    "MIA": 0.93,   # Jeff Hafley
    "CLE": 0.93,   # Todd Monken
    "LV":  0.93,   # Klint Kubiak
    "ARI": 0.93,   # Mike LaFleur
}

# ── PLAYER BIRTH YEARS ───────────────────────────────────────────────────────
# Used to compute each player's 2026 age for the position-specific age-curve penalty.
# Format: player_display_name (exact match) → birth year
# Sources: Pro-Football-Reference, ESPN, Wikipedia player profiles
# Players not listed fall back to: first_season_in_data - avg_draft_age (22)
PLAYER_BIRTH_YEARS: dict[str, int] = {
    # ── Quarterbacks ──────────────────────────────────────────────────────────
    "Aaron Rodgers":      1983,
    "Geno Smith":         1990,
    "Jacoby Brissett":    1992,
    "Kirk Cousins":       1988,
    "Matthew Stafford":   1988,
    "Dak Prescott":       1993,
    "Jared Goff":         1994,
    "Baker Mayfield":     1995,
    "Patrick Mahomes":    1995,
    "Josh Allen":         1996,
    "Sam Darnold":        1997,
    "Daniel Jones":       1997,
    "Kyler Murray":       1997,
    "Tua Tagovailoa":     1997,
    "Jalen Hurts":        1998,
    "Justin Herbert":     1998,
    "Justin Fields":      1999,
    "Tyler Shough":       1999,
    "Lamar Jackson":      1997,
    "Jordan Love":        1998,
    "Will Levis":         2000,
    "Bo Nix":             2000,
    "C.J. Stroud":        2001,
    "Bryce Young":        2001,
    "Malik Willis":       1999,
    "Caleb Williams":     2001,
    "Drake Maye":         2002,
    "Jaxson Dart":        2002,
    "Cam Ward":           2002,
    # ── Running Backs ─────────────────────────────────────────────────────────
    "Austin Ekeler":       1995,
    "Aaron Jones":         1994,
    "Alvin Kamara":        1995,
    "Nick Chubb":          1995,
    "Kareem Hunt":         1995,
    "Joe Mixon":           1996,
    "Derrick Henry":       1994,
    "Christian McCaffrey": 1996,
    "Saquon Barkley":      1997,
    "Tony Pollard":        1997,
    "David Montgomery":    1997,
    "Zack Moss":           1997,
    "D'Andre Swift":       1999,
    "Travis Etienne":      1999,
    "Rhamondre Stevenson": 1998,
    "Brian Robinson":      1998,
    "Josh Jacobs":         1998,
    "Chuba Hubbard":       1999,
    "Jaylen Warren":       1999,
    "Isiah Pacheco":       1999,
    "Rico Dowdle":         1998,
    "Kenneth Gainwell":    1999,
    "Bijan Robinson":      2001,
    "Jonathan Taylor":     1999,
    "James Cook":          2000,
    "Javonte Williams":    2000,
    "Kenneth Walker III":  2000,
    "Kyren Williams":      2000,
    "Dameon Pierce":       2000,
    "Chase Brown":         2001,
    "Tank Bigsby":         2001,
    "Breece Hall":         2001,
    "De'Von Achane":       2002,
    "Jahmyr Gibbs":        2002,
    "RJ Harvey":           2002,
    "TreVeyon Henderson":  2002,
    "Ashton Jeanty":       2003,
    # ── Wide Receivers ────────────────────────────────────────────────────────
    "Davante Adams":          1992,
    "Keenan Allen":           1992,
    "Tyler Lockett":          1992,
    "DeAndre Hopkins":        1992,
    "Odell Beckham":          1992,
    "Mike Evans":             1993,
    "Cooper Kupp":            1993,
    "Stefon Diggs":           1993,
    "Calvin Ridley":          1994,
    "Courtland Sutton":       1995,
    "Terry McLaurin":         1995,
    "Tyreek Hill":            1994,
    "Deebo Samuel Sr.":       1996,
    "DJ Moore":               1997,
    "Diontae Johnson":        1996,
    "Michael Pittman":        1997,
    "DK Metcalf":             1997,
    "A.J. Brown":             1997,
    "Chris Godwin":           1996,
    "Tee Higgins":            1999,
    "Justin Jefferson":       1999,
    "Jaylen Waddle":          1999,
    "Nico Collins":           1999,
    "DeVonta Smith":          1998,
    "Wan'Dale Robinson":      2001,
    "George Pickens":         2001,
    "Drake London":           2001,
    "Jameson Williams":       2001,
    "Rashee Rice":            2001,
    "Puka Nacua":             2001,
    "Michael Wilson":         2001,
    "Khalil Shakir":          2000,
    "Ja'Marr Chase":          2000,
    "CeeDee Lamb":            2000,
    "Garrett Wilson":         2000,
    "Zay Flowers":            2000,
    "Chris Olave":            2000,
    "Amon-Ra St. Brown":      1999,
    "Christian Watson":       1999,
    "Jaxon Smith-Njigba":     2002,
    "Brian Thomas":           2002,
    "Emeka Egbuka":           2002,
    "Tetairoa McMillan":      2003,
    "Marvin Harrison":        2003,
    "Malik Nabers":           2003,
    # ── Tight Ends ────────────────────────────────────────────────────────────
    "Travis Kelce":        1989,
    "Taysom Hill":         1990,
    "Tyler Higbee":        1993,
    "Gerald Everett":      1993,
    "George Kittle":       1993,
    "Hunter Henry":        1994,
    "Evan Engram":         1994,
    "Dallas Goedert":      1995,
    "Mark Andrews":        1995,
    "David Njoku":         1996,
    "T.J. Hockenson":      1997,
    "Pat Freiermuth":      1999,
    "Cade Otton":          1999,
    "Cole Kmet":           1999,
    "Trey McBride":        2000,
    "Jake Ferguson":       2000,
    "Chigoziem Okonkwo":   2000,
    "Kyle Pitts":          2000,
    "Isaiah Likely":       2001,
    "Tucker Kraft":        2001,
    "Sam LaPorta":         2001,
    "Harold Fannin Jr.":   2002,
    "Michael Mayer":       2002,
    "Tyler Warren":        2002,
    "Brock Bowers":        2003,
}


def _age_factor(pos: str, age: int) -> float:
    """Position-specific age-curve multiplier for 2026 projections.

    Research basis (Harvard Sports Analysis, 4for4, Rotoviz aging curves):
    • QB:  Peak 28–32. Gradual decline. Elite QBs can maintain through 35.
           Cliff at 37+ (arm strength, mobility, recovery).
    • RB:  Peak 22–25. Fastest decline of any position — physical toll of
           carries and blitz pickups compounds quickly after age 27.
    • WR:  Peak 24–27. Moderate decline from 28; routes/separation hold
           longer than RB athleticism, but cliff arrives at 32+.
    • TE:  Peak 25–29. Most gradual decline — blocking + receiving split
           means pure athleticism matters less than QB/RB/WR.
    """
    age = int(age)
    if pos == "QB":
        if age <= 24: return 1.02
        if age <= 27: return 1.01
        if age <= 32: return 1.00   # prime years — no adjustment
        if age <= 34: return 0.97
        if age <= 36: return 0.93
        if age <= 38: return 0.88
        return 0.82                  # 39+ (Aaron Rodgers tier)
    elif pos == "RB":
        if age <= 22: return 1.04   # burst-year potential for rookie backs
        if age <= 24: return 1.02
        if age <= 26: return 1.00   # prime years
        if age <= 27: return 0.97
        if age <= 28: return 0.93
        if age <= 29: return 0.88
        if age <= 30: return 0.82
        return 0.74                  # 31+ (severe decline expected)
    elif pos == "WR":
        if age <= 22: return 1.03
        if age <= 24: return 1.02
        if age <= 27: return 1.00   # prime years
        if age <= 29: return 0.98
        if age <= 31: return 0.94
        if age <= 33: return 0.89
        return 0.82                  # 34+
    elif pos == "TE":
        if age <= 23: return 1.02
        if age <= 25: return 1.01
        if age <= 29: return 1.00   # prime years
        if age <= 31: return 0.97
        if age <= 33: return 0.93
        return 0.87                  # 34+
    return 1.0   # unknown position


# ── TEAM OFFENSIVE TIER MULTIPLIERS ──────────────────────────────────────────
# Derived from actual 2025 NFL regular-season data (weekly.csv).
# All 32 teams split evenly into thirds: top 10 / mid 10 / bot 12.
# Research basis: team passing environment explains ~15-20% of QB/WR fantasy
# variance; rushing environment explains ~15% of RB variance.
#
# Top tier (+8%): elite environment lifts all skill-position players
# Mid tier ( 0%): league-average; no adjustment
# Bot tier (-7%): poor environment caps upside despite individual talent

# Passing offense tier — applied to WR and TE ONLY (QBs are exempt — see _tier_mult).
# Ranked by WR+TE combined fantasy PPG per team. This captures the actual receiver
# environment (target volume, QB accuracy, scheme) rather than raw passing yardage,
# which is skewed by mobile QBs (e.g. Josh Allen's rushing means BUF passes less).
# Source: 2025 weekly.csv — WR+TE fantasy_points_ppr per game per team.
PASSING_OFFENSE_TIERS = {
    # ── Top tier (1.08) — ranks 1-10 ─────────────────────────────────────────
    "DET": 1.08,  # 11.36 WR+TE PPG — #1 receiver environment
    "LAR": 1.08,  # 10.34
    "ARI": 1.08,  #  9.94
    "DAL": 1.08,  #  9.91
    "SEA": 1.08,  #  9.56
    "NO":  1.08,  #  9.10
    "PHI": 1.08,  #  9.10
    "CIN": 1.08,  #  9.09
    "IND": 1.08,  #  9.08
    "CHI": 1.08,  #  8.91
    # ── Mid tier (1.00) — ranks 11-20 ────────────────────────────────────────
    "NYG": 1.00,  "TB":  1.00,  "ATL": 1.00,  "NE":  1.00,  "KC":  1.00,
    "SF":  1.00,  "HOU": 1.00,  "LAC": 1.00,  "JAX": 1.00,  "MIN": 1.00,
    # ── Bot tier (0.93) — ranks 21-32 ────────────────────────────────────────
    "BAL": 0.93,  "DEN": 0.93,  "LV":  0.93,  "WAS": 0.93,  "MIA": 0.93,
    "GB":  0.93,  "BUF": 0.93,  "PIT": 0.93,  "CAR": 0.93,  "NYJ": 0.93,
    "CLE": 0.93,  "TEN": 0.93,
}

# Rushing offense tier — applied to RB only
# Ranked by composite: rush yds/gm + (rush TD/gm × 20). Source: 2025 weekly.csv
RUSHING_OFFENSE_TIERS = {
    # ── Top tier (1.06) — ranks 1-10 ─────────────────────────────────────────
    "BUF": 1.06,  # 158.1 ypg / 1.71 TD/g — #1 rushing offense
    "BAL": 1.06,  # 151.4 ypg / 1.24 TD/g
    "CHI": 1.06,  # 136.1 ypg / 1.06 TD/g
    "NYG": 1.06,  # 129.5 ypg / 1.29 TD/g
    "NE":  1.06,  # 127.4 ypg / 1.29 TD/g
    "WAS": 1.06,  # 129.4 ypg / 1.12 TD/g
    "IND": 1.06,  # 114.8 ypg / 1.53 TD/g
    "ATL": 1.06,  # 125.2 ypg / 1.00 TD/g
    "DET": 1.06,  # 117.8 ypg / 1.24 TD/g
    "DAL": 1.06,  # 118.4 ypg / 1.06 TD/g
    # ── Mid tier (1.00) — ranks 11-20 ────────────────────────────────────────
    "PHI": 1.00,  "LAR": 1.00,  "SEA": 1.00,  "JAX": 1.00,  "NYJ": 1.00,
    "DEN": 1.00,  "LAC": 1.00,  "GB":  1.00,  "MIA": 1.00,  "TB":  1.00,
    # ── Bot tier (0.94) — ranks 21-32 ────────────────────────────────────────
    "SF":  0.94,  "CAR": 0.94,  "MIN": 0.94,  "KC":  0.94,  "HOU": 0.94,
    "PIT": 0.94,  "CIN": 0.94,  "NO":  0.94,  "TEN": 0.94,  "ARI": 0.94,
    "CLE": 0.94,  "LV":  0.94,
}

# 2026 projected games overrides — ONLY for players with confirmed game-count limitations.
# Everyone else defaults to NFL_GAMES (17): healthy starters are assumed to play a full season.
# PPG penalties for poor team environments are handled via PASSING/RUSHING_OFFENSE_TIERS.
# Format: player_name_fragment → projected games in 2026
PROJ_GAMES_OVERRIDES = {
    "Rashee Rice":      10,   # NFL suspension; available ~Week 7+ (~10 games projected)
    "Patrick Mahomes":  14,   # ACL recovery; conservative Week 1 availability uncertain
}

# ══════════════════════════════════════════════════════════════════════════════
# RIDGE REGRESSION (pure numpy — no sklearn dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _ridge_fit(X: np.ndarray, y: np.ndarray,
               alpha: float = RIDGE_ALPHA,
               weights: np.ndarray | None = None):
    """Weighted ridge regression via the regularised normal equation."""
    n = X.shape[0]
    Xb = np.column_stack([np.ones(n), X])
    if weights is not None:
        w = weights / weights.sum() * n      # normalise so sum = n
        W = np.diag(w)
        XtW = Xb.T @ W
        A, bv = XtW @ Xb, XtW @ y
    else:
        A, bv = Xb.T @ Xb, Xb.T @ y
    reg = np.eye(A.shape[0]) * alpha
    reg[0, 0] = 0                           # do not regularise the intercept
    theta = np.linalg.solve(A + reg, bv)
    yp = Xb @ theta
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - yp) ** 2)))
    return theta[1:], theta[0], r2, rmse

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

weekly = load_weekly()
teams  = load_teams()

if weekly.empty:
    st.warning("No weekly player data found. Run load_nfl_data.py first.")
    st.stop()

name_col  = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
id_col    = next((c for c in ["player_id", "gsis_id"]                        if c in weekly.columns), None)
team_col  = next((c for c in ["recent_team", "posteam", "team"]              if c in weekly.columns), None)
pos_col   = next((c for c in ["position", "pos"]                             if c in weekly.columns), None)

if not name_col or not pos_col:
    st.error("Required columns (player name, position) not found.")
    st.stop()

track_col = id_col if id_col else name_col

# ── Sidebar ──────────────────────────────────────────────────────────────────
if "pred_v" not in st.session_state:
    st.session_state["pred_v"] = 0
_v = st.session_state["pred_v"]

sel_pos = st.sidebar.selectbox("Position", ["All"] + list(POSITION_FEATURES.keys()),
                               key=f"pred_pos_{_v}")
top_n = st.sidebar.slider("Big Board Size", 10, 200, 100, key=f"pred_top_{_v}")

if st.sidebar.button("Reset Filters", key="pred_reset", use_container_width=True):
    st.session_state["pred_v"] = _v + 1
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building 2026 projections …")
def build_predictions(weekly_df: pd.DataFrame):
    # ── Regular season only ──────────────────────────────────────────────────
    reg = weekly_df.copy()
    if "season_type" in reg.columns:
        reg = reg[reg["season_type"] == "REG"]
    reg = reg[reg[pos_col].isin(POSITION_FEATURES)]

    # ── Season aggregates ────────────────────────────────────────────────────
    all_feat_cols = list({c for feats in POSITION_FEATURES.values() for c in feats})
    stat_cols     = [TARGET_COL] + [c for c in all_feat_cols if c in reg.columns]
    group_keys    = [track_col, name_col, pos_col, "season"]
    if team_col:
        group_keys.append(team_col)

    agg = reg.groupby(group_keys, as_index=False)[stat_cols].sum()
    gp  = reg.groupby(group_keys, as_index=False)[TARGET_COL].count()
    gp.rename(columns={TARGET_COL: "games"}, inplace=True)
    agg = agg.merge(gp, on=group_keys, how="left")

    # Per-game rates (the core features the model trains on)
    for c in stat_cols:
        if c in agg.columns:
            agg[f"{c}_pg"] = (agg[c] / agg["games"].clip(lower=1)).round(4)
    # game_rate encodes starter reliability / injury history
    agg["game_rate"] = (agg["games"] / NFL_GAMES).clip(upper=1.0).round(4)

    all_seasons = sorted(agg["season"].dropna().unique().astype(int))
    latest_szn  = all_seasons[-1]
    prev_szn    = all_seasons[-2] if len(all_seasons) >= 2 else None

    predictions_list = []

    for pos, base_feats in POSITION_FEATURES.items():
        min_g    = MIN_GAMES_BY_POS[pos]
        pg_feats = [f"{c}_pg" for c in base_feats if f"{c}_pg" in agg.columns] + ["game_rate"]

        pos_df = agg[(agg[pos_col] == pos) & (agg["games"] >= min_g)].copy()
        if pos_df.empty:
            continue

        # ── Training: season-N per-game features → season-N+1 total points ──
        train_X, train_y, train_w = [], [], []
        for szn in all_seasons[:-1]:
            nxt = szn + 1
            if nxt not in all_seasons:
                continue
            cur_d = pos_df[pos_df["season"] == szn].set_index(track_col)
            nxt_d = pos_df[pos_df["season"] == nxt].set_index(track_col)
            common = cur_d.index.intersection(nxt_d.index)
            # Recency weight: most-recent season pairs weighted highest
            w = float(np.exp(DECAY * (szn - (latest_szn - 1))))
            for pid in common:
                rc = (cur_d.loc[pid] if isinstance(cur_d.loc[pid], pd.Series)
                      else cur_d.loc[pid].iloc[-1])
                rn = (nxt_d.loc[pid] if isinstance(nxt_d.loc[pid], pd.Series)
                      else nxt_d.loc[pid].iloc[-1])
                train_X.append([float(rc.get(f, 0) or 0) for f in pg_feats])
                train_y.append(float(rn[TARGET_COL]))
                train_w.append(w)

        if len(train_X) < 20:
            continue

        Xtr = np.array(train_X, dtype=np.float64)
        ytr = np.array(train_y, dtype=np.float64)
        wtr = np.array(train_w, dtype=np.float64)

        # Standardise for numerical stability
        mu    = Xtr.mean(0)
        sigma = Xtr.std(0); sigma[sigma == 0] = 1.0
        coefs, intercept, r2, rmse = _ridge_fit((Xtr - mu) / sigma, ytr, weights=wtr)

        # ── Predict on latest season ─────────────────────────────────────────
        lat = pos_df[pos_df["season"] == latest_szn].copy()
        if lat.empty:
            continue

        Xlat     = (lat[pg_feats].fillna(0).values.astype(np.float64) - mu) / sigma
        raw_pred = np.clip(Xlat @ coefs + intercept, 0, None)

        # ── Project 2026 games played ────────────────────────────────────────
        # All healthy starters are assumed to play the full 17-game season.
        # Specific exceptions (suspensions, confirmed carry-over injuries) are
        # applied post-model via PROJ_GAMES_OVERRIDES.
        # games_lat is retained to drive the adj_factor: per-game efficiency from
        # an injury-shortened 2025 still needs scaling to a full 17-game season.
        games_lat  = lat["games"].values.astype(float)
        proj_games = np.full(len(lat), float(NFL_GAMES))   # 17 by default

        # adj_factor: scale raw_pred (trained on season totals) to 17 games.
        # Damped at 0.45 so a player who only played 4 games isn't over-extrapolated.
        # Capped at 1.10 so partial-season starters (e.g. 13-game QBs) can't be
        # inflated by more than 10% — otherwise a 13-game QB gets a 13.8% boost vs
        # a 16-game elite QB's 2.8%, creating a structural bias against full-season stars.
        adj_factor = np.clip(
            1.0 + (proj_games / games_lat.clip(min=1) - 1.0) * 0.45,
            a_min=None, a_max=1.10
        )
        model_pts = np.clip(raw_pred * adj_factor, 0, None)

        # ── QB-specific: recency-weighted multi-year PPG × 17 games ─────────
        # QBs are projected on true fantasy value assuming full health (17g).
        # Injury risk is tracked separately via the injury_risk flag.
        # Weights: most recent season 65%, previous year 25%, two years ago 10%.
        # Research shows the most recent season explains ~65% of next-year QB
        # variance — this weighting is more responsive than DECAY=0.35 (44%).
        if pos == "QB":
            # Explicit season weights: most-recent=65%, 2nd-most-recent=25%, 3rd=10%
            # Older seasons are excluded (weight 0) — only last 3 seasons matter.
            _szn_sorted = sorted(all_seasons)   # ascending: oldest first
            _n = len(_szn_sorted)
            # Build weight map: position from END (0=most recent, 1=prev, 2=two years ago)
            _recency_w = {0: 0.65, 1: 0.25, 2: 0.10}
            _szn_weight = {
                s: _recency_w.get(_n - 1 - i, 0.0)   # 0.0 for seasons older than 3 years
                for i, s in enumerate(_szn_sorted)
            }

            wtd_ppg = np.zeros(len(lat))
            wtd_sum = np.zeros(len(lat))
            for szn in all_seasons:
                szn_df = pos_df[pos_df["season"] == szn].set_index(track_col)
                w_szn  = _szn_weight.get(szn, 0.10)
                for idx, row in lat.iterrows():
                    pid = row[track_col]
                    if pid in szn_df.index:
                        ppg_szn = szn_df.loc[pid, f"{TARGET_COL}_pg"] if isinstance(
                            szn_df.loc[pid], pd.Series) else szn_df.loc[pid].iloc[-1][f"{TARGET_COL}_pg"]
                        if ppg_szn and ppg_szn > 0:
                            lat_idx = lat.index.get_loc(idx)
                            wtd_ppg[lat_idx] += float(ppg_szn) * w_szn
                            wtd_sum[lat_idx] += w_szn
            wtd_sum = np.where(wtd_sum == 0, 1.0, wtd_sum)
            qb_weighted_ppg = wtd_ppg / wtd_sum
            # Fall back to prior-year PPG for QBs with no historical data
            fallback_ppg = lat[f"{TARGET_COL}_pg"].fillna(0).values
            qb_weighted_ppg = np.where(qb_weighted_ppg > 0, qb_weighted_ppg, fallback_ppg)
            pred_pts = np.clip(qb_weighted_ppg * float(NFL_GAMES), 0, None)
        else:
            # ── Prior-year PPG baseline (45% weight) for RB/WR/TE ────────────
            # Anchor projections to last season's actual per-game output.
            # Uses PPG_BASELINE_GAMES (position-specific, < 17) instead of the
            # full proj_games to correct for systematic over-prediction:
            # backtesting showed projecting 17 games for the PPG anchor creates
            # +44-pt QB bias and +10-pt RB bias — because real starters average
            # 13-16 games due to injuries, bye weeks, and load management.
            # The model component still projects to 17 games (proj_games);
            # only the PPG anchor is calibrated to the realistic expected games.
            prior_ppg      = lat[f"{TARGET_COL}_pg"].fillna(0).values
            ppg_proj_g     = float(PPG_BASELINE_GAMES.get(pos, NFL_GAMES))
            ppg_baseline   = np.clip(prior_ppg * ppg_proj_g, 0, None)
            pred_pts = (model_pts * (1.0 - PPG_BLEND_WEIGHT)
                        + ppg_baseline * PPG_BLEND_WEIGHT)

        lat = lat.copy()
        lat["predicted_pts"] = pred_pts.round(1)
        lat["proj_games"]    = proj_games.round(1)
        lat["pred_ppg"]      = (pred_pts / proj_games.clip(min=1)).round(2)
        lat["rmse"]          = round(rmse, 1)

        # ── Injury risk flag ──────────────────────────────────────────────────────
        # QBs: historical games-played average < 14.5 over last 3 years
        # RB/WR/TE: expert injury mapping (NFL research as of March 2026)
        # Projection assumes healthy 17-game season regardless of flag.
        if pos == "QB":
            avg_g_map: dict = {}
            for szn in all_seasons[-3:]:
                szn_g = pos_df[pos_df["season"] == szn].set_index(track_col)["games"]
                for pid, g in szn_g.items():
                    avg_g_map.setdefault(pid, []).append(float(g))
            lat["injury_risk"] = lat[track_col].map(
                lambda pid: "      Yes      " if (
                    len(avg_g_map.get(pid, [])) > 0 and
                    sum(avg_g_map.get(pid, [17])) / len(avg_g_map.get(pid, [17])) < 14.5
                ) else ""
            )
        else:
            # RB/WR/TE: check expert injury risk map
            lat["injury_risk"] = lat[name_col].map(
                lambda player_name: INJURY_RISK_MAP.get(player_name, "")
            )

        predictions_list.append(lat)

    if not predictions_list:
        return pd.DataFrame(), pd.DataFrame()

    all_preds = pd.concat(predictions_list, ignore_index=True)

    # Historical per-season totals for the trajectory line chart (include games for PPG tooltip)
    hist = (agg.groupby([track_col, name_col, pos_col, "season"], as_index=False)
               [[TARGET_COL, "games"]].sum())
    if team_col and team_col in agg.columns:
        tl = agg.groupby([track_col, "season"], as_index=False)[team_col].first()
        hist = hist.merge(tl, on=[track_col, "season"], how="left")

    return all_preds, hist


all_preds_raw, hist_totals = build_predictions(weekly)


def apply_expert_adjustments(df: pd.DataFrame,
                              raw_weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply NFL Expert 2026 roster corrections on top of the statistical model."""
    if df.empty:
        return df
    out = df.copy()

    # 1. Remove players not projected as 2026 starters
    out = out[~out[name_col].isin(EXPERT_REMOVE)].copy()

    # 2. Deduplicate (keep highest predicted_pts per player/position)
    out = out.sort_values("predicted_pts", ascending=False).drop_duplicates(
        subset=[name_col, pos_col], keep="first"
    )

    # 3. Force-inject confirmed starters filtered out by injury-shortened seasons
    if raw_weekly is not None and not raw_weekly.empty:
        reg_w = raw_weekly.copy()
        if "season_type" in reg_w.columns:
            reg_w = reg_w[reg_w["season_type"] == "REG"]

        for player_name, (player_id, pos, team_2026, manual_ppg) in FORCE_INCLUDE_STARTERS.items():
            already_in = out[name_col].str.contains(player_name, case=False, na=False).any()
            if already_in:
                continue

            p_data = reg_w[reg_w[name_col] == player_name].copy()
            min_g = MIN_GAMES_BY_POS.get(pos, 6)

            if manual_ppg is not None:
                # Player has no qualifying historical season (e.g. career backup turned starter).
                # Use the expert-supplied PPG directly with a full projected-games estimate.
                ppg      = float(manual_ppg)
                proj_g   = float(MAX_PROJ_GAMES)   # assume full-season starter
                proj_pts = round(ppg * proj_g, 1)
                games_2025 = int(p_data[p_data["season"] == PREDICTION_YEAR - 1][TARGET_COL].count()
                                 if not p_data.empty else 0)
                actual_2025 = float(p_data[p_data["season"] == PREDICTION_YEAR - 1][TARGET_COL].sum()
                                    if not p_data.empty else 0)
                display_games = games_2025 if games_2025 > 0 else min_g
            else:
                if p_data.empty:
                    continue
                # All seasons aggregated (used for weighted PPG and fallback)
                p_seas = (p_data.groupby("season")[TARGET_COL]
                          .agg(games="count", total_pts="sum")
                          .reset_index())
                # Need at least one qualifying season to anchor the projection
                qualifying = p_seas[p_seas["games"] >= min_g].sort_values("season", ascending=False)
                if qualifying.empty:
                    continue
                best = qualifying.iloc[0]

                # Recency-weighted multi-year PPG — same approach as main QB model.
                # Include ALL seasons with 5+ games so injury-shortened years
                # (e.g. Daniels 7g in 2025 at 16.33 PPG) are weighted in properly.
                # Without this, FORCE_INCLUDE QBs only use their best healthy season
                # and ignore evidence of decline or volatility.
                _szn_weights = {s: (1.0 + DECAY) ** i
                                for i, s in enumerate(sorted(p_seas["season"].unique()))}
                usable = p_seas[p_seas["games"] >= 5]
                if not usable.empty:
                    _wtd = sum(
                        (float(r["total_pts"]) / float(r["games"])) * _szn_weights.get(int(r["season"]), 1.0)
                        for _, r in usable.iterrows()
                    )
                    _w_sum = sum(_szn_weights.get(int(r["season"]), 1.0) for _, r in usable.iterrows())
                    ppg = _wtd / _w_sum if _w_sum > 0 else (float(best["total_pts"]) / float(best["games"]))
                else:
                    ppg = float(best["total_pts"]) / float(best["games"])

                games_2025 = int(p_seas[p_seas["season"] == PREDICTION_YEAR - 1]["games"].sum()
                                 if (PREDICTION_YEAR - 1) in p_seas["season"].values else 0)
                # Project all force-include starters at full 17 games — they are confirmed
                # starters; the injury risk flag already communicates the health caveat.
                proj_g    = float(NFL_GAMES)
                proj_pts  = round(ppg * proj_g, 1)
                actual_2025   = float(p_data[p_data["season"] == PREDICTION_YEAR - 1][TARGET_COL].sum())
                display_games = games_2025 if games_2025 > 0 else float(best["games"])

            new_row: dict = {
                name_col:      player_name,
                pos_col:       pos,
                "season":      PREDICTION_YEAR - 1,
                "games":       display_games,
                TARGET_COL:    round(actual_2025, 1) if actual_2025 > 0 else round(float(best["total_pts"]), 1),
                "predicted_pts": proj_pts,
                "proj_games":  proj_g,
                "pred_ppg":    round(ppg, 2),
                "rmse":        0.0,
                "injury_risk": (
                    ("      Yes      " if float(display_games) < 14.5 else "")
                    if pos == "QB" else
                    INJURY_RISK_MAP.get(player_name, "")
                ),
            }
            if track_col != name_col:
                new_row[track_col] = player_id
            if team_col:
                new_row[team_col] = team_2026

            out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)

    # 4. Team corrections (trades / FA signings not captured in historical data)
    if team_col:
        for player_fragment, new_team in EXPERT_TEAM_CORRECTIONS.items():
            mask = out[name_col].str.contains(player_fragment, case=False, na=False)
            out.loc[mask, team_col] = new_team

    # 5. Team offensive tier multipliers (position-aware, based on 2025 data)
    #    Applied AFTER team corrections so trades/FA moves use the correct 2026 team.
    #    WR + TE  → PASSING_OFFENSE_TIERS  (receiver environment / target volume)
    #    QB       → exempt (1.0) — QB skill IS what drives team passing stats;
    #               applying a team passing tier to QBs is circular and incorrectly
    #               penalises elite mobile QBs (e.g. Josh Allen) whose team passing
    #               yardage ranks low because they contribute heavily via rushing.
    #    RB       → RUSHING_OFFENSE_TIERS
    if team_col and pos_col:
        recv_pos  = {"WR", "TE"}
        rushing_pos = {"RB"}
        def _tier_mult(row):
            team = row.get(team_col, "")
            pos  = row.get(pos_col, "")
            if pos in recv_pos:
                return PASSING_OFFENSE_TIERS.get(team, 1.0)
            elif pos in rushing_pos:
                return RUSHING_OFFENSE_TIERS.get(team, 1.0)
            return 1.0  # QB and all other positions: no team passing tier
        mults = out.apply(_tier_mult, axis=1)
        out["predicted_pts"] = (out["predicted_pts"] * mults).round(1)
        out["pred_ppg"]      = (out["pred_ppg"]      * mults).round(2)

    # 6. New head coach penalty — applied to ALL positions on affected teams
    #    Stacked on top of offensive tier multipliers (both apply independently)
    if team_col:
        hc_mults = out[team_col].map(lambda t: NEW_HC_PENALTY.get(t, 1.0))
        out["predicted_pts"] = (out["predicted_pts"] * hc_mults).round(1)
        out["pred_ppg"]      = (out["pred_ppg"]      * hc_mults).round(2)

    # 7. Age-curve penalty — position-specific, derived from 2026 player age
    #    Applied last so all other corrections (team, HC) already reflect true
    #    2026 context before age multiplies in.
    #    For players not in PLAYER_BIRTH_YEARS, fallback = first_season - 22
    #    (average NFL draft age across all skill positions).
    if pos_col and name_col:
        _AVG_DRAFT_AGE = {"QB": 22, "RB": 22, "WR": 22, "TE": 22}

        # Build first-season lookup from raw weekly data (fallback)
        _first_szn: dict = {}
        if raw_weekly is not None and not raw_weekly.empty:
            _raw_reg = raw_weekly
            if "season_type" in raw_weekly.columns:
                _raw_reg = raw_weekly[raw_weekly["season_type"] == "REG"]
            _first_szn = _raw_reg.groupby(name_col)["season"].min().to_dict()

        def _player_age_2026(name: str, pos: str) -> int:
            if name in PLAYER_BIRTH_YEARS:
                return 2026 - PLAYER_BIRTH_YEARS[name]
            first = _first_szn.get(name, 2020)
            return 2026 - (first - _AVG_DRAFT_AGE.get(pos, 22))

        age_mults = out.apply(
            lambda r: _age_factor(
                str(r.get(pos_col, "")),
                _player_age_2026(str(r.get(name_col, "")), str(r.get(pos_col, "")))
            ),
            axis=1
        )
        out["predicted_pts"] = (out["predicted_pts"] * age_mults).round(1)
        out["pred_ppg"]      = (out["pred_ppg"]      * age_mults).round(2)

    return out.reset_index(drop=True)


all_preds = apply_expert_adjustments(all_preds_raw, weekly)


def apply_games_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply confirmed 2026 game-count reductions for suspensions / carry-over injuries.

    All other players have already been projected at NFL_GAMES (17) inside build_predictions.
    This function only touches players in PROJ_GAMES_OVERRIDES.
    Predicted points are recalculated as  pred_ppg × new_proj_games  so the per-game
    efficiency (already adjusted by team tier multipliers) is preserved exactly.
    """
    out = df.copy()
    for player_fragment, games in PROJ_GAMES_OVERRIDES.items():
        mask = out[name_col].str.contains(player_fragment, case=False, na=False)
        if mask.any():
            out.loc[mask, "proj_games"]    = float(games)
            out.loc[mask, "predicted_pts"] = (out.loc[mask, "pred_ppg"] * games).round(1)
    return out.reset_index(drop=True)


all_preds = apply_games_overrides(all_preds)


def _assign_vor(df: pd.DataFrame) -> pd.DataFrame:
    """Add VOR, ADP, ADP round, and draft value columns.

    VOR = predicted_pts − replacement_level[position]
    Replacement level is calibrated to a 10-team PPR league (10 picks per round) based on 2025
    championship team analysis. Sorting by VOR rather than raw points accounts
    for positional scarcity — an elite TE ranks higher than an equivalent-points RB.

    ADP columns:
      adp       — consensus overall pick number (10-team PPR, April 2026)
      adp_round — draft round derived from ADP (e.g. pick 23 → Rd 3)
      value     — ADP pick − model rank. Positive = model ranks player higher
                  than consensus (undervalued steal). Negative = ADP overrates them.
    """
    out = df.copy()

    # ── VOR ──────────────────────────────────────────────────────────────────
    out["vor"] = out.apply(
        lambda r: round(float(r["predicted_pts"]) - REPLACEMENT_LEVEL.get(r[pos_col], 0), 1),
        axis=1,
    )

    # ── Model rank (1 = best by VOR across all positions) ────────────────────
    out = out.sort_values("vor", ascending=False).reset_index(drop=True)
    out["_model_rank"] = range(1, len(out) + 1)

    # ── ADP lookup ───────────────────────────────────────────────────────────
    out["adp"] = out[name_col].apply(get_adp)

    # adp_round: round number for a 10-team league (10 picks per round)
    out["adp_round"] = out["adp"].apply(
        lambda x: int((x - 1) // 10) + 1 if x < 999 else None
    )
    # Display as "Rd N" string; unranked players show "—"
    out["adp_round"] = out["adp_round"].apply(
        lambda x: f"Rd {x}" if pd.notna(x) else "—"
    )

    # adp display: show pick number or "—" for unranked
    out["adp_display"] = out["adp"].apply(
        lambda x: f"{x:.0f}" if x < 999 else "—"
    )

    # value: positive = model likes them more than ADP (undervalued)
    out["value"] = out.apply(
        lambda r: int(round(r["adp"] - r["_model_rank"])) if r["adp"] < 999 else None,
        axis=1,
    )

    out = out.drop(columns=["_model_rank"])
    return out


all_preds = _assign_vor(all_preds)

if all_preds.empty:
    st.error("Not enough historical data to build predictions.")
    st.stop()

# ── Filter by position ───────────────────────────────────────────────────────
preds = (all_preds[all_preds[pos_col] == sel_pos].copy()
         if sel_pos != "All" else all_preds.copy())
# Sort by VOR (positional scarcity-adjusted value) rather than raw points
preds = preds.sort_values("vor", ascending=False).head(top_n).reset_index(drop=True)
preds.insert(0, "Rank", range(1, len(preds) + 1))

# Delta vs last season
preds["last_season_pts"] = preds[TARGET_COL].round(1)
preds["last_season_ppg"] = (preds["last_season_pts"] / preds["games"].replace(0, np.nan)).round(2).fillna(0.0)
preds["change"]          = (preds["predicted_pts"] - preds["last_season_pts"]).round(1)
preds["change_pct"]      = ((preds["change"] / preds["last_season_pts"].replace(0, np.nan)) * 100).round(1)

if preds.empty:
    st.info("No predictions available for the selected position.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TOP 3 HERO CARDS  — single markdown block avoids Streamlit column-height overlap
# ══════════════════════════════════════════════════════════════════════════════

pos_label_str = f"({sel_pos})" if sel_pos != "All" else "(Overall)"
st.markdown(f"### 🏆 2026 Projected Top 3 {pos_label_str}")

if len(preds) >= 3:
    cards_html = '<div style="display:flex;gap:16px;align-items:stretch;margin-bottom:8px;">'
    for i, medal in enumerate(["🥇 #1", "🥈 #2", "🥉 #3"]):
        row         = preds.iloc[i]
        player      = row[name_col]
        team_abbr   = row[team_col] if team_col else "—"
        pos_lbl     = row[pos_col] if pos_col else ""
        pred_pts    = row["predicted_pts"]
        pred_ppg    = row["pred_ppg"]
        proj_g      = row["proj_games"]
        delta       = row["change"]
        delta_sign  = "+" if delta >= 0 else ""
        delta_color = "#10b981" if delta >= 0 else "#ef4444"
        url         = get_logo(team_abbr, teams)
        logo_html   = f'<img src="{url}" width="40" style="margin:6px 0 4px;">' if url else ""
        cards_html += f"""
        <div class="stat-card" style="flex:1;min-width:0;text-align:center;">
            <div class="label">{medal}</div>
            {logo_html}
            <div class="value" style="font-size:1.1rem;line-height:1.3;word-break:break-word;">{player}</div>
            <div class="sub">{team_abbr} &nbsp;·&nbsp; {pos_lbl}</div>
            <div style="font-size:1.45rem;font-weight:800;color:#f59e0b;margin:8px 0 2px;">
                {pred_pts:,.1f}<span style="font-size:0.78rem;font-weight:500;color:#8b8fa8;">&thinsp;proj pts</span>
            </div>
            <div class="sub">{pred_ppg} PPG &nbsp;·&nbsp; {proj_g:.0f} games projected</div>
            <div class="sub" style="margin-top:4px;">
                <span style="color:{delta_color};font-weight:600;">{delta_sign}{delta:,.1f}</span>
                <span style="color:#8b8fa8;"> vs last season</span>
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2026 BIG BOARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### 📋 2026 Fantasy Big Board {pos_label_str}")

board_cols = ["Rank", name_col]
if team_col: board_cols.append(team_col)
if pos_col:  board_cols.append(pos_col)
board_cols += ["injury_risk", "predicted_pts", "vor", "adp_display", "adp_round", "value", "pred_ppg", "proj_games", "last_season_pts", "change", "change_pct", "games", "last_season_ppg"]

# Position-specific counting stats
if sel_pos in ("QB", "All"):
    for c in ["passing_yards", "passing_tds", "interceptions", "rushing_yards"]:
        if c in preds.columns and c not in board_cols:
            board_cols.append(c)
if sel_pos in ("RB", "All"):
    for c in ["rushing_yards", "rushing_tds", "carries", "receptions", "receiving_yards"]:
        if c in preds.columns and c not in board_cols:
            board_cols.append(c)
if sel_pos in ("WR", "TE", "All"):
    for c in ["receiving_yards", "receiving_tds", "targets", "receptions"]:
        if c in preds.columns and c not in board_cols:
            board_cols.append(c)

board_cols = [c for c in board_cols if c in preds.columns]

rename_map = {
    name_col: "Player",
    "injury_risk": "Injury Risk",
    "predicted_pts": "2026 Proj",
    "vor":          "VOR",
    "adp_display":  "ADP",
    "adp_round":    "ADP Rd",
    "value":        "Value",
    "pred_ppg": "Proj PPG",
    "proj_games": "Proj GP",
    "last_season_pts": "2025 Actual",
    "last_season_ppg": "2025 PPG",
    "change": "Δ Pts",
    "change_pct": "Δ %",
    "games": "2025 GP",
    "passing_yards": "Pass Yds", "passing_tds": "Pass TD", "interceptions": "INT",
    "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD", "carries": "Carries",
    "receptions": "Rec",   "targets": "Tgt",
    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
}
if team_col: rename_map[team_col] = "Team Abb"
if pos_col:  rename_map[pos_col]  = "Pos"

disp = preds[board_cols].copy()
# Ensure injury_risk never shows "None" — blank for non-risk players
if "injury_risk" in disp.columns:
    disp["injury_risk"] = disp["injury_risk"].fillna("").replace({None: "", "None": ""})
for c in disp.select_dtypes("float").columns:
    if c != "change_pct":
        disp[c] = disp[c].round(1)

# Add logo URLs for team column if it exists
teams_df = load_teams()
column_config_dict = {
    "Injury Risk": st.column_config.TextColumn(
                      label="Injury Risk",
                      width="small",
                      help="Yes = Injury risk (avg < 14.5 games/yr over last 3 seasons). "
                           "Blank = durable starter. "
                           "Projections assume full 17-game season regardless."),
    "2026 Proj":  st.column_config.NumberColumn(format="%.1f"),
    "VOR":        st.column_config.NumberColumn(format="%.1f",
                      help="Value Over Replacement — positional scarcity-adjusted score. "
                           "Accounts for how scarce elite players are at each position "
                           "(QB=240, RB=115, WR=95, TE=80 replacement baselines)."),
    "ADP":        st.column_config.TextColumn(
                      help="2026 consensus Average Draft Position (10-team PPR, April 2026). "
                           "Overall pick number. '—' = unranked / undrafted."),
    "ADP Rd":     st.column_config.TextColumn(
                      help="Draft round derived from ADP (10 picks per round)."),
    "Value":      st.column_config.NumberColumn(
                      format="%+d",
                      help="ADP pick − model rank. "
                           "Positive = model ranks this player higher than consensus ADP (undervalued steal). "
                           "Negative = consensus ADP is more bullish than this model (potential avoid). "
                           "Blank = unranked in ADP."),
    "Proj PPG":   st.column_config.NumberColumn(format="%.2f"),
    "2025 PPG":   st.column_config.NumberColumn(format="%.2f"),
    "Δ Pts":      st.column_config.NumberColumn(format="%+.1f"),
    "Δ %":        st.column_config.NumberColumn(format="%+.1f%%"),
}

if team_col:
    disp["_logo_url"] = disp[team_col].apply(lambda t: get_logo(t, teams_df) if pd.notna(t) else "")
    column_config_dict["_logo_url"] = st.column_config.ImageColumn(
        label="Team",
        width="small",
    )

disp_renamed = disp.rename(columns=rename_map)
if "_logo_url" in disp.columns:
    # Reorder to put logo before Team
    cols = list(disp_renamed.columns)
    if "_logo_url" in cols:
        cols.remove("_logo_url")
        cols.insert(cols.index("Team Abb"), "_logo_url")
    disp_renamed = disp_renamed[cols]


st.dataframe(
    disp_renamed,
    hide_index=True,
    use_container_width=True,
    column_config=column_config_dict,
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RISERS & FALLERS — per-position % change so QBs don't dominate raw-point deltas
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Risers &nbsp;&nbsp; 📉 Fallers")
st.caption("Ranked by % change within each position — QBs and skill positions compared fairly on relative improvement. (Top 200 players only)")

positions_to_show = [sel_pos] if sel_pos != "All" else list(POSITION_FEATURES.keys())

# Only show risers/fallers for players in the top 200 overall (ranked by VOR)
top_200_df = all_preds.nlargest(200, "vor").reset_index(drop=True).copy()
top_200_df.insert(0, "overall_rank", range(1, len(top_200_df) + 1))
top_200_board = top_200_df[name_col].tolist()
rank_map = dict(zip(top_200_df[name_col], top_200_df["overall_rank"]))

for pos in positions_to_show:
    pos_preds = all_preds[(all_preds[pos_col] == pos) & (all_preds[name_col].isin(top_200_board))].copy()
    pos_preds["last_season_pts"] = pos_preds[TARGET_COL].round(1)
    pos_preds["change"]          = (pos_preds["predicted_pts"] - pos_preds["last_season_pts"]).round(1)
    pos_preds["change_pct"]      = ((pos_preds["change"] /
                                     pos_preds["last_season_pts"].replace(0, np.nan)) * 100).round(1)
    pos_preds = pos_preds.dropna(subset=["change_pct"])

    if pos_preds.empty:
        continue

    rise_cols = [name_col, team_col, "predicted_pts", "last_season_pts", "change", "change_pct"]
    rise_cols = [c for c in rise_cols if c in pos_preds.columns]

    # Get risers and fallers
    risers_df  = pos_preds.nlargest(5,  "change_pct")[rise_cols].copy()
    fallers_df = pos_preds.nsmallest(5, "change_pct")[rise_cols].copy()

    # Add overall rank column
    risers_df["overall_rank"]  = risers_df[name_col].map(rank_map)
    fallers_df["overall_rank"] = fallers_df[name_col].map(rank_map)

    rise_rename = {name_col: "Player", team_col: "Team",
                   "predicted_pts": "2026 Proj", "last_season_pts": "2025 Pts",
                   "change": "Δ Pts", "change_pct": "Δ %", "overall_rank": "Rank"}

    risers  = risers_df.rename(columns=rise_rename)
    fallers = fallers_df.rename(columns=rise_rename)

    # Add logo URLs for team column
    rise_col_config = {"Δ Pts": st.column_config.NumberColumn(format="%+.1f"),
                       "Δ %":   st.column_config.NumberColumn(format="%+.1f%%"),
                       "Rank":  st.column_config.NumberColumn(format="%d")}

    if "Team" in risers.columns:
        risers["_logo_url"] = risers["Team"].apply(lambda t: get_logo(t, teams_df) if pd.notna(t) else "")
        fallers["_logo_url"] = fallers["Team"].apply(lambda t: get_logo(t, teams_df) if pd.notna(t) else "")

        rise_col_config["_logo_url"] = st.column_config.ImageColumn(label="", width="small")

        # Reorder to put logo before Team
        risers_cols = list(risers.columns)
        fallers_cols = list(fallers.columns)
        for cols_list in [risers_cols, fallers_cols]:
            if "_logo_url" in cols_list:
                cols_list.remove("_logo_url")
                cols_list.insert(cols_list.index("Team"), "_logo_url")
        risers = risers[risers_cols]
        fallers = fallers[fallers_cols]

    st.markdown(f"**{POSITION_LABELS[pos]}**")
    # Tabs give each table full width → no canvas blur that occurs inside narrow st.columns
    tab_r, tab_f = st.tabs(["📈 Risers", "📉 Fallers"])
    with tab_r:
        st.dataframe(risers, hide_index=True,
                     use_container_width=True,
                     column_config=rise_col_config)
    with tab_f:
        st.dataframe(fallers, hide_index=True,
                     use_container_width=True,
                     column_config=rise_col_config)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "**Methodology** — Position-specific ridge regression trained on 2016–2025 consecutive-season pairs "
    "with exponential recency weighting (recent seasons count more). Features are per-game rates, not "
    "season totals, so a player who missed games due to injury is not penalised for low counting stats. "
    "Projected 2026 games blends the last two seasons (65 / 35 weighting) with a conservative ceiling of "
    f"{MAX_PROJ_GAMES} games. QB qualifier: {MIN_GAMES_BY_POS['QB']}+ games started. Skill positions: 6+ games. "
    "**VOR (Value Over Replacement)** ranks players by positional scarcity in a 10-team league: "
    "elite TEs rank higher than equivalent-point WRs because only 10 starting TEs exist. Replacement levels "
    f"(QB={REPLACEMENT_LEVEL['QB']}, RB={REPLACEMENT_LEVEL['RB']}, WR={REPLACEMENT_LEVEL['WR']}, "
    f"TE={REPLACEMENT_LEVEL['TE']}) calibrated from 2025 championship team data. "
    "**ADP (Average Draft Position)** — 2026 consensus overall pick number from FantasyPros/Sleeper "
    "community data (10-team PPR, April 2026). ADP Rd is the draft round (10 picks/round). "
    "**Value** = ADP pick − model rank: a +15 means the model projects this player 15 spots better "
    "than consensus ADP, making them a target in drafts; a −10 means consensus is more bullish "
    "than the model projects. Players not yet ranked in consensus ADP show '—'. "
    "**Expert overlays** applied post-model using live 2026 offseason data: team corrections "
    "(Kyler→MIN, Waddle→DEN, DJ Moore→BUF, Pittman→PIT, Walker→KC, Evans→SF, Etienne→NO, "
    "Henry→BAL, Darnold→SEA, Keenan Allen→LAC, Hopkins→BAL, Dowdle→PIT, Pickens→DAL), "
    "removals (Penix—ACL, Tua—ATL backup, Kamara demoted, Ekeler—Achilles, Nabers—ACL), "
    "force-includes (Kyler Murray—5 games 2025, Willis—MIA starter, Shough—NO starter, Stafford—LAR returning), "
    "age-cliff discounts (Kelce 0.82×, Evans 0.80×, CMC 0.92×), "
    "injury/suspension cuts (Rice 0.70×, Mahomes 0.92×), "
    "and breakout boosts (Gibbs 1.22×, Skattebo 1.20×, JSN 1.18×, Irving 1.18×, Pickens 1.10×, Pitts 1.10×, Jefferson 1.08×)."
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY LINE CHART — historical + projected 2026
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Historical Trajectory + 2026 Projection")

default_players = preds.head(5)[name_col].tolist()
chart_players = st.multiselect(
    "Select players to chart",
    options=preds[name_col].tolist(),
    default=default_players,
    key=f"pred_chart_{_v}",
)

if chart_players:
    fig = go.Figure()
    for player in chart_players:
        pr = preds[preds[name_col] == player]
        if pr.empty:
            continue
        pid      = pr.iloc[0][track_col]
        t_abbr   = pr.iloc[0][team_col] if team_col else ""
        color    = TEAM_COLORS.get(t_abbr, "#4f46e5")
        pred_val = float(pr.iloc[0]["predicted_pts"])

        ph = hist_totals[hist_totals[track_col] == pid].sort_values("season")
        if ph.empty:
            continue
        szns = ph["season"].tolist()
        vals = ph[TARGET_COL].tolist()
        ppgs = (ph[TARGET_COL] / ph["games"].clip(lower=1)).round(2).tolist()

        fig.add_trace(go.Scatter(
            x=szns, y=vals, mode="lines+markers", name=player,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(color="#fff", width=1)),
            customdata=list(zip(ppgs, ph["games"].tolist())),
            hovertemplate=(
                f"<b>{player}</b><br>"
                "%{x}: %{y:,.1f} pts<br>"
                "PPG: %{customdata[0]:.2f} · %{customdata[1]:.0f} games"
                "<extra></extra>"
            ),
            legendgroup=player,
        ))
        pred_ppg_val = float(pr.iloc[0]["pred_ppg"])
        fig.add_trace(go.Scatter(
            x=[szns[-1], PREDICTION_YEAR], y=[vals[-1], pred_val],
            mode="lines+markers", showlegend=False, legendgroup=player,
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=10, color=color, symbol="star",
                        line=dict(color="#fff", width=1.5)),
            hovertemplate=(
                f"<b>{player}</b><br>"
                f"2026 Proj: {pred_val:,.1f} pts<br>"
                f"Proj PPG: {pred_ppg_val:.2f} · 17 games"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Fantasy PPR Points — Historical + 2026 Projection",
        xaxis_title="Season", yaxis_title="Total Fantasy Points (PPR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(dtick=1)
    fig.add_vrect(
        x0=hist_totals["season"].max() + 0.5, x1=PREDICTION_YEAR + 0.5,
        fillcolor="#4f46e5", opacity=0.05, layer="below", line_width=0,
        annotation_text="Projected", annotation_position="top left",
        annotation_font_color="#888",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one player to see their trajectory.")
