from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo, load_depth_charts
from utils.nav import render_sidebar_nav, render_last_updated
from utils.tables import UNRANKED_MARK, rank_display
from model.projection import (
    ProjectionConfig, build_predictions_core, derive_replacement_baseline,
    NFL_GAMES, POSITION_FEATURES, MIN_GAMES_BY_POS,
    RIDGE_ALPHA, DECAY, PPG_BLEND_WEIGHT, PPG_BASELINE_GAMES,
    _weighted_durability, _expected_games,
    AVAIL_SHRINK, AVAIL_GAMES_FLOOR, AVAIL_GAMES_CEILING, AVAIL_RECENCY_WEIGHTS,
)

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

# Every curated input the projections are built from. The stamp shows the newest
# of them, so hand-editing any expert file is reflected on the page immediately.
# This module counts as one of those inputs: the injury flags, games overrides and
# player multipliers below are hand-maintained here, not in data/, and editing them
# moves the numbers just as much as a new CSV does.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
render_last_updated(__file__, *(RAW_DIR / f for f in (
    "weekly.csv", "rookies_2026.csv", "kickers_2026.csv",
    "defenses_2026.csv", "espn_ranks_2026.csv")), label="Projections updated")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# TARGET_COL is reassigned dynamically based on the sidebar scoring selector
# (PPR / Half PPR / Standard). The default below is used only during initial
# module load — the value is overwritten before build_predictions() runs.
TARGET_COL = "fantasy_points_ppr"
PREDICTION_YEAR = 2026
DEFAULT_PROJ_GAMES = 14.0   # fallback when prior-season data is absent

# Map sidebar scoring selection → underlying column in weekly.csv.
# Half PPR is derived on the fly: (fantasy_points + fantasy_points_ppr) / 2.
SCORING_TARGET_COLS = {
    "PPR":      "fantasy_points_ppr",
    "Half PPR": "fantasy_points_half_ppr",   # derived below
    "Standard": "fantasy_points",
}

# NFL_GAMES, MIN_GAMES_BY_POS, RIDGE_ALPHA, DECAY, PPG_BLEND_WEIGHT, PPG_BASELINE_GAMES,
# and POSITION_FEATURES now live in model/projection.py (imported above) — that module
# is the single source of truth so the Streamlit page and scripts/backtest_model.py can
# never drift apart on the projection engine's own constants.
MAX_PROJ_GAMES = 16       # conservative ceiling (no one is guaranteed 17) — K/DEF + fallback

# ── Sanity guards on the final per-game rate ────────────────────────────────
# Two conservative corrections applied AFTER every contextual multiplier, using
# each player's real game log:
#   • Peak cap: never project a player above 105% of their best qualifying
#     season PPG — stops boosts from stacking a player past their own ceiling
#     (e.g. George Pickens projected above his career-best 2025).
#   • Low-sample regression: shrink players with few career games toward a
#     typical established starter at their position, so a hot 8-game stretch
#     (e.g. Cam Skattebo) can't drive a top ranking. The anchor is the median
#     of established players projected above replacement level (a real starter,
#     not a replacement scrub), and the pull is Weight = games / (games + K).
PEAK_CAP_MULT      = 1.05  # cap projected PPG at 105% of best real season
PEAK_MIN_GAMES     = 6     # a season needs ≥ this many games to count as a peak
REGRESS_MIN_GAMES  = 24    # players with fewer career games get shrunk to baseline
REGRESS_K          = 10    # pseudo-games of positional baseline in the shrink

# Players exempt from the peak cap. The cap assumes a player's best season is a
# talent ceiling, which holds for anyone who has already had a full workload —
# but NOT for a career backup who just won a starting job. Their peak measures
# the old ROLE, so capping there silently deletes the promotion. Everyone here
# set their previous high while splitting snaps; the 2026 job is a new one.
PEAK_CAP_EXEMPT = {
    "Isaiah Likely",       # career high came as the BAL TE2 behind Andrews; now the NYG TE1
    "Darnell Washington",  # rotational blocker in PIT; now the starter on a 4-yr/$42M deal
    "Chig Okonkwo",        # TEN committee TE; now the WAS TE1 inheriting the Ertz targets
    "Kenneth Walker III",  # SEA carries were split with Charbonnet; KC signed him to be the bell cow
    "Bhayshul Tuten",      # rookie year was a committee behind Etienne; now the JAX lead back
    "Jahmyr Gibbs",        # every career high was set splitting DET carries with David Montgomery,
                           #   who is in HOU for 2026. The peak prices the committee, not the job.
    "Matthew Golden",      # rookie year was the GB WR4 at 5.4 PPG behind Doubs/Reed/Watson;
                           #   Doubs left for NE, so the peak measures a role he no longer has.
}

# ── Value Over Replacement (VOR) — positional scarcity scoring ───────────────
# Replacement level = projected points of the LAST startable player at each
# position in a 10-team PPR league (i.e. the worst starter on opening day).
# SEED values only. _assign_vor overwrites QB/RB/WR/TE in place with baselines
# derived from the board's own projections (derive_replacement_baseline), so the
# numbers below are not what VOR ends up using. They still matter for two things:
#   • K and DEF, which are never derived — see the sentinel note on those entries.
#   • The coarse "is this a starter" PPG threshold in the regression step, which
#     runs BEFORE projections are final and so can't use the derived baseline.
# Originally calibrated from weekly.csv 2024–2025 actual season finishes:
#     2024 PPR:  QB10=297.2  RB24=191.8  WR30=196.5  TE10=163.3
#     2025 PPR:  QB10=286.9  RB24=179.4  WR30=175.8  TE10=177.7
# That calibration is exactly what made these unusable as a VOR baseline: actual
# finishes come off ~16.5 games played while projections sit on a ~13-game
# baseline, so subtracting one from the other mixed units.
REPLACEMENT_LEVEL = {
    "QB":  290,   # ~QB10 average; QB punt strategy still viable but cliff is steeper than prior 240
    "RB":  185,   # ~RB24 average (2 starters + flex in a 10-team league)
                  # NOTE: QB/RB/WR/TE entries here are only seed values — _assign_vor
                  # overwrites them with baselines derived from the live projection
                  # pool at the current LEAGUE_SIZE. Only K/DEF survive as sentinels.
    "WR":  185,   # ~WR30 average (3 starters + flex)
    "TE":  170,   # ~TE10 average; elite TEs (Kelce-tier) still command a premium
    "K":   220,   # deliberately inflated (well above any kicker projection) so
                  # kicker VOR is strongly negative and they sort into the final
                  # rounds — mirrors real drafts where kickers go last
    "DEF": 200,   # same idea for team defenses: inflated above every D/ST
                  # projection so their VOR is strongly negative and they land
                  # in the final rounds (DST is streamed, drafted late)
}

# Per-scoring-format replacement levels. Half-PPR averages standard + PPR.
# Standard PPR (no per-reception bonus) drops WR/TE replacement floors materially.
# K and DEF scoring is format-independent, so their floors are constant.
SCORING_REPLACEMENT_LEVELS = {
    "PPR":      {"QB": 290, "RB": 185, "WR": 185, "TE": 170, "K": 220, "DEF": 200},
    "Half PPR": {"QB": 290, "RB": 175, "WR": 150, "TE": 140, "K": 220, "DEF": 200},
    "Standard": {"QB": 290, "RB": 160, "WR": 115, "TE": 105, "K": 220, "DEF": 200},
}

# LEAGUE SIZE — used to derive round grades from model rank (picks per round = league size)
# and to set how deep the replacement baseline sits. Confirmed against the league's
# own ESPN settings payload for 2026 (settings.size == 11), which expanded from 10.
LEAGUE_SIZE = 11

# Roster shape. Drives the replacement baseline in _assign_vor: replacement is
# whoever sits just past the last startable player at each position, so it moves
# with league settings. In superflex, for example, the QB baseline collapses and
# elite QBs correctly climb into the early rounds — something a hardcoded
# constant can never follow.
ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

POSITION_LABELS = {"QB": "Quarterbacks", "RB": "Running Backs",
                   "WR": "Wide Receivers",  "TE": "Tight Ends", "K": "Kickers",
                   "DEF": "Defenses"}

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
    "Tyler Shough":     ("00-0040743", "QB", "NO",  15.8),   # NO QB1; his actual 2025 rate (157.96 pts / 10 g) held over a full season — older-than-typical rookie, so no ascending-QB boost
    "Matthew Stafford": ("00-0026498", "QB", "LAR", None),   # Returning for 2026 with LAR; find most recent qualifying season
    "Jayden Daniels":   ("00-0039910", "QB", "WAS", None),   # 7 games 2025 (injury); uses 2024 full season (20.93 PPG)
    "Malik Nabers":     ("00-0039337", "WR", "NYG", None),   # 4 games 2025 (ACL); confirmed NYG WR1 for 2026; uses 2024 full season
    "James Conner":     ("00-0033553", "RB", "ARI", None),   # 3 games 2025 (ankle); reworked deal to stay ARI as backup behind rookie Love; uses 2024 full season
    "Joe Burrow":       ("00-0036442", "QB", "CIN", None),   # 8 games 2025 (turf toe); healthy CIN QB1, consensus QB4 — uses 2024 full season
    "Brock Purdy":      ("00-0037834", "QB", "SF",  None),   # 11 games 2025 (toe/shoulder); confirmed SF QB1 — uses 2024 season
    "Kirk Cousins":     ("00-0029604", "QB", "LV",  None),   # 10 games 2025; named LV QB1 to open camp over rookie Mendoza — uses 2024 season
    # ── Aug-24-2026 audit: starters the min-games gate dropped off the board ──
    # Each missed most/all of 2025 with an injury, so their LATEST season fails
    # MIN_GAMES_BY_POS and build_predictions_core never scores them — but all are
    # healthy, rostered 2026 starters that ESPN ranks inside its positional lists.
    "Jayden Reed":      ("00-0039146", "WR", "GB",  None),   # 5 games 2025 (collarbone/foot); GB slot WR1 — uses 2024 full season
    "Tank Dell":        ("00-0038977", "WR", "HOU", None),   # missed all of 2025 (knee); cleared for camp — uses 2024 season
    "Jalen McMillan":   ("00-0039855", "WR", "TB",  None),   # 4 games 2025 (hamstring); TB WR3 — uses 2024 season
    "Jonathon Brooks":  ("00-0039344", "RB", "CAR", 9.0),    # two ACLs, 3 career games; CAR change-of-pace back behind Hubbard — expert PPG
    "Najee Harris":     ("00-0036893", "RB", "NYG", 7.5),    # 3 games 2025 (Achilles); signed NYG Aug-2026 and is
                                                             #   already the RB2 ahead of Singletary and Tracy. His
                                                             #   2021-24 seasons all qualify, so the automatic path
                                                             #   would price the PIT bell-cow role he no longer has —
                                                             #   this is a handcuff rate behind a shaky Skattebo
    # ── Aug-30-2026: the back Jacobs' exempt-list placement promotes ──────────
    # One career game across two seasons (hamstring, then IR), so MIN_GAMES_BY_POS
    # drops him entirely and the board had GB's Week 1 starter missing. Beat
    # reporting has him as the favourite for the largest share while Jacobs is
    # away. The rate is a blend, not a lead-back rate: roughly seven games at
    # ~12 PPR as the committee lead, then ~4 once Jacobs is back.
    "MarShawn Lloyd":   ("00-0039811", "RB", "GB",  8.0),
}

# Players removed from 2026 board (not projected starters / retired / injury risk)
EXPERT_REMOVE = {
    "Rob Gronkowski",    # Officially retired March 2026
    "Michael Penix",     # ACL surgery (Nov 2025); still not medically cleared for 11-on-11 in Aug-2026 camp
                         #   while Tua takes every first-team rep — not the projected Week 1 ATL starter
    "Jayden Higgins",    # Torn ACL in the Aug-2026 joint practice vs LV — out for the 2026 season
    "Austin Ekeler",     # Torn Achilles; out for 2026 season
    "Zach Ertz",         # Unsigned FA (2026); no confirmed team — removed pending signing
    "Zavier Scott",      # MIN RB4/practice-squad-caliber; buried behind the Jones/Mason committee, not fantasy-relevant
    "Brandin Cooks",     # Unsigned FA (2026) as of late July; no confirmed team — removed pending signing
    "Ricky Pearsall",    # OUT FOR 2026 — PCL surgery announced Aug 2026 (chronic since Wk4 2025)
}

# Injury risk mapping — 2026 season outlook
# Based on NFL Expert research as of March 2026. Applied to ALL positions.
# QBs: checked in addition to the historical games-played average auto-flag.
# RB/WR/TE: primary injury risk source.
# Flag criteria: significant injury in 2024-25 OR documented recurring injury history.
# "      Yes      " padding is intentional — centers text in the narrow column.
INJURY_RISK_MAP = {

    # ── QUARTERBACKS ─────────────────────────────────────────────────────────
    # (auto-flag also triggers if avg games/yr < 14.5 over last 3 seasons)
    "Patrick Mahomes":    "      Yes      ",   # ACL/knee surgery (2025); proj 14 games 2026; historically durable but confirmed injury concern
    "Jordan Love":        "      Yes      ",   # Elbow injury (2024); missed 4 games; elbow injuries notorious for QB re-injury risk
    "C.J. Stroud":        "      Yes      ",   # Burner/nerve injury (2024); missed games; shoulder/nerve recurrence risk
    "Jalen Hurts":        "      Yes      ",   # Shoulder injuries (2022); missed games (2024); recurring soft-tissue profile
    "Dak Prescott":       "      Yes      ",   # Achilles tear (2020, missed season); hamstring 2023; chronic injury profile
    "Daniel Jones":       "      Yes      ",   # Torn Achilles Wk14 2025; confirmed IND starter but Wk1 availability uncertain
    "Joe Burrow":         "      Yes      ",   # Turf toe surgery (2025, 8 games); wrist 2023 — recurring availability risk
    "Brock Purdy":        "      Yes      ",   # Toe/shoulder (2025); missed 6 games

    # ── RUNNING BACKS ────────────────────────────────────────────────────────
    # Returning from significant 2024/2025 injury
    "Isiah Pacheco":      "      Yes      ",   # Fractured fibula (Dec 2024); missed final stretch of season
    "De'Von Achane":      "      Yes      ",   # Knee/hamstring (multiple 2024); IR stint; limited durability track record

    # Chronic / well-documented injury history
    "Christian McCaffrey":"      Yes      ",   # Extensive history: ribs (2020), hamstring/ankle/calf (multiple seasons), IR 2025
    "Breece Hall":        "      Yes      ",   # ACL tear Week 7 (2022 rookie year); fully recovered but carries re-injury risk
    "Jonathan Taylor":    "      Yes      ",   # Ankle injuries (2023); recurring soft tissue; missed multiple games
    "Travis Etienne":     "      Yes      ",   # Lisfranc surgery (2021, missed entire rookie season); hip/foot issues 2024
    "Javonte Williams":   "      Yes      ",   # ACL/LCL/MCL tear Week 4 (2022); high-risk ligament profile
    "Kyren Williams":     "      Yes      ",   # ACL tear (2022 rookie year); ankle issues 2024; structural re-injury concern
    "D'Andre Swift":      "      Yes      ",   # Shoulder/hip injuries (2021-2023); missed significant time each year
    "Josh Jacobs":        "      Yes      ",   # Quad/hamstring soft tissue issues, and as of Aug-30-2026 on the
                                               #   commissioner's exempt list — the availability risk is off-field
    "Rhamondre Stevenson":"      Yes      ",   # Ankle injuries (2022, 2024); recurring lower-body issues
    "Kenneth Walker III": "      Yes      ",   # Hernia surgery (2023); oblique 2024; missed multiple games
    "Brian Robinson":     "      Yes      ",   # Gunshot wound recovery (2022); shoulder/knee issues since
    "Jaylen Warren":      "      Yes      ",   # Shoulder (2023); hamstring (2024); limited full-season history
    "Tony Pollard":       "      Yes      ",   # High ankle sprain (2023 playoffs); hamstring issues with TEN
    "Joe Mixon":          "      Yes      ",   # Ankle surgery (2023); multiple injury stints throughout career
    "Nick Chubb":         "      Yes      ",   # Torn ACL/MCL/PCL (2023, full season); ACL re-injury risk remains elevated
    "Aaron Jones":        "      Yes      ",   # Knee/MCL injuries (2023); recurring soft tissue
    "Dameon Pierce":      "      Yes      ",   # Hamstring (2023); ankle (2024); limited availability both seasons
    "Alvin Kamara":       "      Yes      ",   # Sprained MCL (Aug 2026, 4-6 wks) on top of an injury-plagued 2025; age 31
    "Kyle Monangai":      "      Yes      ",   # Hyperextended right knee (Aug 2026); week-to-week, Wk1 in doubt
    "Ashton Jeanty":      "      Yes      ",   # Wk1 availability in doubt (Aug 2026) on top of a 300+ touch rookie
                                               #   workload — the LV offense has no second bell-cow to absorb it

    # ── WIDE RECEIVERS ───────────────────────────────────────────────────────
    # Returning from significant 2024/2025 injury
    "Malik Nabers":       "      Yes      ",   # ACL tear (Oct 2025); missed rest of season; confirmed NYG WR1 for 2026
    "Brandon Aiyuk":      "      Yes      ",   # ACL/MCL tear (2025); missed entire season
    "Tank Dell":          "      Yes      ",   # Multi-ligament (ACL/MCL/LCL/meniscus); missed 2025
    "Christian Watson":   "      Yes      ",   # ACL tear (Jan 2025); significant 2026 miss risk
    "Puka Nacua":         "      Yes      ",   # Knee injury (2024); missed majority of season
    "Rashee Rice":        "      Yes      ",   # May-2026 knee surgery + off-field risk (suspension served); full-go for camp
    "Chris Godwin Jr.":   "      Yes      ",   # ACL (2021); ankle dislocation/fracture (Oct 2024); chronic injury profile
    "Stefon Diggs":       "      Yes      ",   # ACL (2024, 8 games); played full 2025, but age 33 + Aug-2026 signing = zero camp ramp with WAS
    "Tyreek Hill":        "      Yes      ",   # Multi-ligament knee injury (ACL+); released by MIA
    "Nico Collins":       "      Yes      ",   # Hamstring tear (2024); missed final 6 games of regular season
    "Jordyn Tyson":       "      Yes      ",   # Hamstring (~2 months, Aug 2026) after separate hamstring trouble in the spring
    "Emeka Egbuka":       "      Yes      ",   # Sprained toe (Aug 2026); TB "optimistic" for Wk1 but Bowles stayed non-committal

    # Chronic / recurring history
    "Justin Jefferson":   "      Yes      ",   # Hamstring tear (2023); missed 8 games; recurring hamstring risk
    "CeeDee Lamb":        "      Yes      ",   # Shoulder injury (2024); concussion history; missed games
    "A.J. Brown":         "      Yes      ",   # Knee injury (2024); missed 3 games; chronic knee concern throughout career
    "Tee Higgins":        "      Yes      ",   # Hamstring/ribs (2022-2024); missed games every season
    "Cooper Kupp":        "      Yes      ",   # ACL (2022); ankle/hamstring history (2023-2025); persistent injury profile
    "Jaylen Waddle":      "      Yes      ",   # Ankle/shoulder (2024); concussion 2023; recurring soft tissue
    "Calvin Ridley":      "      Yes      ",   # Mental health leave (2021, full season); hamstring/knee issues 2023-2024
    "Courtland Sutton":   "      Yes      ",   # ACL tear (2020, full season); various soft tissue since
    "Deebo Samuel Sr.":   "      Yes      ",   # Shoulder surgery (2022); rib fracture/hamstring (2023-2024); chronic injury profile
    "Chris Olave":        "      Yes      ",   # Concussion protocol (2023, missed 5 games); documented concussion risk
    "Jameson Williams":   "      Yes      ",   # ACL tear (college 2021); hamstring (2023); recurring injury concern
    "Wan'Dale Robinson":  "      Yes      ",   # ACL tear (2022 rookie year); limited healthy seasons
    "Keenan Allen":       "      Yes      ",   # Chronic hamstring injuries throughout career; age 33 elevates risk
    "DeAndre Hopkins":    "      Yes      ",   # Knee surgery (2023); age 33; transition to BAL adds load uncertainty
    "Diontae Johnson":    "      Yes      ",   # Recurring hamstring issues (2022-2024); missed games each year
    "Michael Pittman":    "      Yes      ",   # Various injuries (2025); new team (PIT) adds uncertainty

    # ── TIGHT ENDS ───────────────────────────────────────────────────────────
    # Returning from significant 2024/2025 injury
    "George Kittle":      "      Yes      ",   # Torn Achilles (Jan 2026); opened camp on Active/PUP, Reserve/PUP still possible
    "Sam LaPorta":        "      Yes      ",   # Back surgery (Nov 2024); recurring back injury risk
    "Tucker Kraft":       "      Yes      ",   # ACL (2025); activated off PUP Jul-2026 but carries ramp-up risk
    "T.J. Hockenson":     "      Yes      ",   # ACL tear (2023); missed entire 2024 season; returning in 2026

    # Chronic / recurring history
    "Dalton Kincaid":     "      Yes      ",   # Recurring PCL issues; missed 5 games in 2025
    "Mark Andrews":       "      Yes      ",   # Ankle fracture + ligament (2023); missed 5 games; recurring ankle concern
    "Dallas Goedert":     "      Yes      ",   # Shoulder/hamstring injuries (2022-2024); missed games every year
    "Kyle Pitts":         "      Yes      ",   # Knee injury (2022); missed 8 games; recurring soft tissue concern
    "Evan Engram":        "      Yes      ",   # Multiple knee/ankle injuries throughout career (NYG era and beyond)
    "Darren Waller":      "      Yes      ",   # Age 34; hamstring/knee history, 7 games in 2025 — has cleared 12 games once since 2021
}

# Team corrections: player name fragment → corrected 2026 team abbreviation
# Sources: ESPN / NFL.com free agency trackers, March 2026
EXPERT_TEAM_CORRECTIONS = {
    "Travis Etienne":    "NO",   # Signed with New Orleans Saints (left JAX)
    "Tua Tagovailoa":    "ATL",  # Signed 1 yr with ATL after his MIA release. Stefanski has NOT formally
                                 #   named a Week 1 starter, but Penix is uncleared (ACL) and Tua has taken
                                 #   every first-team rep, so he is the projected QB1.
    "Kyler Murray":      "MIN",  # 1-year deal with Vikings
    "Jaylen Waddle":     "DEN",  # Traded MIA → DEN (pairs with Bo Nix)
    "Michael Pittman":   "PIT",  # Traded IND → PIT
    "DJ Moore":          "BUF",  # Traded CHI → BUF (Josh Allen boost)
    "Malik Willis":      "MIA",  # 3-yr $67.5M / $45M guaranteed — CONFIRMED MIA starter
    "Kenneth Walker":    "KC",   # 3-yr $43M deal — joins Kansas City
    "Mike Evans":        "SF",   # 3-yr $60M deal — joins 49ers
    "Derrick Henry":     "BAL",  # Re-signed with Baltimore Ravens
    "Sam Darnold":       "SEA",  # Signed with Seattle Seahawks
    "Keenan Allen":      "IND",  # Signed with Indianapolis — WR3 behind Pierce/Downs. Was listed
                                 #   LAC here on a reported return that never happened; he is absent
                                 #   from every 2026 depth chart, so only this entry can place him
    "DeAndre Hopkins":   "BAL",  # Signed with Baltimore Ravens; pairs with Lamar
    "Rico Dowdle":       "PIT",  # Signed with Pittsburgh Steelers (was DAL)
    "Tyler Allgeier":    "ARI",  # Signed 2-yr/$12.25M with Cardinals (was ATL)
    "Elijah Moore":      "PHI",  # Signed with Eagles (March 2026), after BUF release/DEN practice squad stint
    "Jalen Tolbert":     "MIA",  # Signed 1-yr FA deal with Dolphins (left DAL); reunites w/ QB Malik Willis
    "George Pickens":    "DAL",  # Traded PIT → DAL; pairs with Dak Prescott
    "Tyler Shough":      "NO",   # Confirmed New Orleans Saints starter 2026
    # ── 2026 top-150 audit: wrong-team corrections (moves not in historical data)
    "A.J. Brown":        "NE",   # Traded PHI → New England
    "Wan'Dale":          "TEN",  # Signed with Tennessee Titans (left NYG)
    "Rashid Shaheed":    "SEA",  # Signed with Seattle Seahawks (left NO)
    "Kenneth Gainwell":  "TB",   # Signed with Tampa Bay Buccaneers (left PIT)
    "Romeo Doubs":       "NE",   # Signed with New England Patriots (left GB)
    "Chris Rodriguez":   "JAX",  # Signed with Jacksonville Jaguars (left WAS)
    "Jauan Jennings":    "MIN",  # Signed with Minnesota Vikings (left SF)
    "Deebo Samuel":      "SF",   # Re-signed with the 49ers (Aug 2026, 1 yr)
    # ── 2026 TE audit: wrong-team corrections ──────────────────────────────
    "Isaiah Likely":     "NYG",  # Signed 3-yr/$40M with the Giants — follows John Harbaugh, projected TE1
    "Chig Okonkwo":      "WAS",  # Signed 3-yr/$27M with Washington — TE1 replacing Zach Ertz
    "David Njoku":       "LAC",  # Signed 1-yr/$8M with the Chargers (left CLE)
    "Charlie Kolar":     "LAC",  # Signed 3-yr/$24.3M with the Chargers (left BAL)
    "Noah Fant":         "NO",   # Signed with New Orleans — TE2 behind Juwan Johnson
    "Austin Hooper":     "ATL",  # Returns to Atlanta as TE2/3 behind Pitts
    "Daniel Bellinger":  "TEN",  # Signed with Tennessee (left NYG)
    "Johnny Mundt":      "PHI",  # Signed with Philadelphia as TE2 (left JAX)
    # ── Aug-2026 audit: late free-agency moves ─────────────────────────────
    "Stefon Diggs":      "WAS",  # Signed 1-yr/up-to-$12M with Washington (Aug 5, 2026) — WR2 opposite McLaurin
    "Kirk Cousins":      "LV",   # Named Raiders QB1 to open camp over rookie Mendoza
    "Darren Waller":     "CAR",  # Signed with Carolina (Aug 2026) after leaving MIA — TE1 ahead of Tremble
    "Kaleb Johnson":     "GB",   # Traded PIT → GB on Aug 30, 2026, hours after Jacobs went on the exempt
                                 #   list. depth_charts.csv still has him as the PIT RB3 behind Warren/Dowdle
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

# ── PLAYER-SPECIFIC MULTIPLIERS ───────────────────────────────────────────────
# Applied AFTER team-tier, HC, and age-curve adjustments. These capture player
# context the statistical model cannot infer from prior-season stats alone:
#   • Breakout candidates: usage trends, depth-chart promotions, target-share
#     trajectories from PFF / FantasyPros / Establish The Run consensus.
#   • Decline / risk discounts: age cliffs already partially handled by
#     _age_factor, but specific medical situations (Achilles, ACL recovery)
#     get an additional discount here.
#   • Suspension / availability: applied as a season-long multiplier rather
#     than a games-played adjustment, since the underlying PPG stays intact.
#
# Values are calibrated to match the methodology caption shown at the bottom
# of the predictions page. Sources: PFF projections, 4for4 xFP, FantasyPros
# ECR (Mar–Aug 2026 consensus), Matthew Berry's Fantasy Life player tiers.
#
# Keep this list short and high-confidence. Do NOT add players without an
# explicit reason — the team tier / HC / age multipliers carry most of the load.
PLAYER_MULTIPLIERS: dict[str, float] = {
    # ── Breakout boosts (1.08–1.22) ────────────────────────────────────────
    "Jahmyr Gibbs":         1.22,   # Bell-cow workload after Monty departure
    "Cam Skattebo":          0.70,   # NYG lead back, but the committee got worse, not better: his 2025 carry
                                     #   share was 42%, preseason usage did not project a workhorse, and NYG
                                     #   signed Najee Harris in Aug-2026 to take the RB2 snaps. Was 0.82,
                                     #   which still had him RB8 off an 8-game rookie sample (15.7 PPG) —
                                     #   ESPN has him 56th overall, an RB2 rather than a back-end RB1
    "Jaxon Smith-Njigba":    1.00,   # Was 1.18 for "target share trending past Lockett/Metcalf", but
                                     #   Metcalf was traded to PIT after 2024 and JSN then posted 21.2
                                     #   PPG as the outright alpha in 2025. The thesis already played
                                     #   out inside the history the model reads, so the boost was
                                     #   double-counting it — and PEAK_CAP was silently discarding it.
    "Bucky Irving":          0.86,   # Was 1.18 for "Mayfield offense leans on dual-threat RB" — but that
                                     #   receiving work is exactly what TB signed Gainwell to take. OC Zac
                                     #   Robinson has floated a balanced / hot-hand split with Gainwell as
                                     #   the pass-catching back. Irving still leads the carries, so keep the
                                     #   offense boost and drop the dual-threat premium. Games stay at the
                                     #   model's 13.5 (shoulder surgery, injury-hit 2025) even though he is
                                     #   now full-go and has carried in the preseason (TB).
                                     #   NOTE: PEAK_CAP binds here — his pre-cap rate is 18.74 PPG against a
                                     #   cap of 15.10, so any multiplier above ~0.86 is invisible on the
                                     #   board. 0.86 is the largest value that actually moves him.
    "George Pickens":        1.00,   # Was 1.10 for "Dak elevates target quality vs PIT", but he played
                                     #   2025 in Dallas and went 11.7 -> 17.1 PPG doing it. The Dak
                                     #   upgrade is already in the data; the boost re-applied a move
                                     #   that had happened. This is the case PEAK_CAP was built for.
    "Kyle Pitts":            1.00,   # Franchise-tagged then 3-yr/$53M to stay ATL, but QB instability caps him; consensus TE7 vs model TE3
    "Justin Jefferson":      1.20,   # Still a 28.5% target share; model had him WR16 vs consensus WR6 — QB upgrade in MIN
    # ── Veteran decline / age cliffs (0.80–0.92) ───────────────────────────
    "Travis Kelce":          0.95,   # Age 37 but re-signed KC 1-yr/$12M and remains the TE1; consensus TE10, model had him TE~20
    "Mike Evans":            1.05,   # Age 33 and SF competition, but consensus WR19 — model buried him at WR~50
    "Christian McCaffrey":   0.92,   # Coming off Achilles + age-29 curve
    # ── Injury / suspension cuts (0.70–0.92) ───────────────────────────────
    "Rashee Rice":           0.92,   # Suspension served; mild haircut for May-2026 knee surgery + off-field risk
    "Patrick Mahomes":       0.92,   # OL concerns + WR group still maturing
    "Malik Nabers":          0.95,   # Was 0.85 on a "no target date" report; he is now in 11-on-11 at full speed
                                     #   and NYG expect him Wk1 — small haircut left for the non-contact ramp
    # ── Committee / role demotions (0.70–0.85) ─────────────────────────────
    "Tyrone Tracy Jr.":      0.30,   # NYG RB4. Was 0.62 for "top handcuff", which the Aug-2026 preseason
                                     #   erased: he missed a pass-pro assignment vs MIN, fumbled untouched
                                     #   vs MIA, and NYG then signed Najee Harris — the refreshed depth
                                     #   chart has him fourth, behind Skattebo, Harris and Singletary. He
                                     #   is not even the handcuff any more, so the 2025 line (160.8 pts,
                                     #   all of it earned while Skattebo was hurt) describes a role that no
                                     #   longer exists. ESPN projects ~45 points and does not rank him
    "David Montgomery":      1.50,   # HOU lead back. The engine had him RB46 / Rd 15 — below Woody Marks,
                                     #   the RB2 listed under him — because his 2025 line was a DET committee
                                     #   split with Gibbs and HOU carries a weaker rushing tier than DET, so
                                     #   correcting the team actually pushed him further down. Neither effect
                                     #   knows he inherited the job. Sized to ESPN's RB24
    "Bhayshul Tuten":        1.40,   # JAX lead back — same failure as Montgomery. His only NFL season was a
                                     #   15-game rookie committee (88.6 pts) behind Travis Etienne, who is
                                     #   gone; the depth chart now lists Tuten first with Chris Rodriguez Jr.
                                     #   as the change-of-pace. Scoring the old role left him RB37 against a
                                     #   consensus RB25. Sized to ESPN's 62nd overall
    "Chuba Hubbard":         1.18,   # CAR RB1 on the depth chart, but graded RB44 — behind backups on teams
                                     #   with clearer starters — because 2025 was a 15-game split and CAR
                                     #   carries a weak rushing tier that discounts him twice. Jonathon
                                     #   Brooks (0 games, second ACL) is the RB2, so the job is Hubbard's
                                     #   until proven otherwise. Sized to ESPN's 101st overall
    # ── 2026 top-150 audit: same-team committee / role cuts ────────────────
    "Travis Etienne":        1.10,   # Was 0.95 for a pure NO committee; Kamara's Aug-19 MCL sprain (4-6 wks) hands
                                     #   him the lead-back reps through at least the opening month
    "Alvin Kamara":          0.85,   # NO true committee split w/ Etienne — age-31 decline also applies via age curve
    "RJ Harvey":             0.72,   # DEN RB committee (Dobbins/Estime); early-down/passing split
    "Rhamondre Stevenson":   0.85,   # NE committee w/ Henderson; goal-line + volume not guaranteed
    "Kyle Monangai":         0.95,   # CHI RB2, but 0.80 for a "spot starter only" understates a Ben Johnson
                                     #   split — he handles the early-down work and is going around RB34
    "Parker Washington":     0.82,   # JAX WR3 behind Thomas/Hunter target hierarchy
    "Juwan Johnson":         0.85,   # NO TE sharing with rookie Delp; TD-dependent
    "Woody Marks":           0.68,   # HOU RB2. Was 0.78 for a "committee behind Mixon" that no longer
                                     #   exists — Mixon is off the roster and Montgomery signed to lead it.
                                     #   Marks graded out RB31 to Montgomery's RB46, i.e. ahead of the
                                     #   starter listed above him; this puts him back at ESPN's RB42
    "Michael Wilson":        0.85,   # ARI WR3 behind MHJ + Harrison target share
    "Marvin Harrison Jr.":   0.90,   # ARI target share capped by McBride + rookie WR draft
    "Kimani Vidal":          0.75,   # LAC RB2 behind Hampton
    "Bam Knight":            0.35,   # 2025 rate (10.3 ppg/9g) came filling in for injured Conner; buried behind Love/Conner/Allgeier in 2026
    "James Conner":          0.55,   # ARI backup behind rookie Love (reworked deal to stay, but ceded starter role)
    "Tyler Allgeier":        0.65,   # ARI room is crowded, but he opened camp atop the first depth chart ahead of Love
    "Jeremiyah Love":        1.20,   # ARI rookie 3rd overall; real 1A/1B split w/ Allgeier caps him, but consensus RB13-16 vs model RB25
    "Devin Neal":            0.62,   # NO rookie-year committee, 3rd on depth chart
    "Chimere Dike":          0.75,   # TEN WR in crowded young group
    "Elic Ayomanor":         0.72,   # TEN WR2/3 competing for targets
    "Khalil Shakir":         0.88,   # BUF slot but target share diluted by additions
    "Troy Franklin":         0.88,   # DEN WR3 (2025 WR31 finish) but Waddle arrival caps ceiling
    "Quentin Johnston":      0.85,   # LAC WR2 behind McConkey; heavy TD-dependence in 2025 = regression risk
    "Courtland Sutton":      1.00,   # DEN clear WR1 (2025 WR14 finish); Waddle competition already priced into base
    "Travis Hunter":         0.78,   # JAX two-way snap load caps offensive volume
    "DeMario Douglas":       0.78,   # NE slot; target share diluted by roster adds
    # ── 2026 top-150 audit: departure boosts (target share vacated) ────────
    "Rome Odunze":           1.00,   # CHI WR1 after D.J. Moore left, but rookie Luther Burden III eats into the vacated share
    "Emeka Egbuka":          1.12,   # TB target share up with Mike Evans gone to SF
    "Josh Downs":            1.10,   # IND slot volume up after Pittman departure
    "Brock Bowers":          1.25,   # LV featured TE; "100%" after the 2025 PCL/bone bruise, new OC Kubiak building around him — consensus TE1
    "DeVonta Smith":         1.10,   # PHI WR1 target share up after A.J. Brown departure
    "Matthew Golden":        1.08,   # GB target share up after Romeo Doubs departure
    # ── 2026 top-150 audit: injury haircuts (paired w/ games override) ─────
    "Chris Godwin Jr.":      0.88,   # TB ankle recovery (PUP watch), ramp-up expected
    "Alec Pierce":           0.90,   # IND ankle; deep-role volatility
    "Zach Charbonnet":       0.65,   # SEA ACL recovery; limited early-season role
    # ── 2026 TE audit: role changes from free agency / depth-chart moves ────
    "Isaiah Likely":         1.50,   # NYG TE1 on a 3-yr/$40M deal — base rate is a career of ~40% snaps behind Andrews
    "Chig Okonkwo":          1.12,   # WAS TE1 (3-yr/$27M) inheriting the Ertz target share
    "Mark Andrews":          1.35,   # BAL role expands with Likely gone; re-signed 3-yr/$39.2M
    "Darnell Washington":    1.50,   # PIT starter after a 4-yr/$42M extension; base is a career of rotational snaps
    "Pat Freiermuth":        0.78,   # PIT TE2 behind Washington after the extension
    "Theo Johnson":          0.55,   # NYG TE2 behind Likely — lost the starting job
    "Jake Tonges":           0.60,   # SF TE2 again once Kittle returns; 2025 rate came from Kittle's absence
    "Oronde Gadsden II":     0.80,   # LAC 3-way TE split w/ Njoku + Kolar under new OC
    "David Njoku":           0.72,   # LAC committee TE; no longer a featured TE1
    "Charlie Kolar":         0.70,   # LAC TE3 in a crowded room
    "Noah Fant":             0.70,   # NO TE2 behind Juwan Johnson
    "Austin Hooper":         0.62,   # ATL TE2/3 behind Kyle Pitts
    "Mason Taylor":          0.65,   # NYJ TE2, not co-starter — ADP fell from ~150 to past 250 once Sadiq went 16th overall
    "Colby Parkinson":       0.75,   # LAR 5-deep TE committee
    "Tyler Higbee":          0.72,   # LAR committee; age-33 snap management
    "Terrance Ferguson":     0.60,   # LAR room jammed w/ Higbee, Parkinson, Klare — not draftable in most formats

    # ══ AUG-2026 SOFT AUDIT (top-25 per position vs camp reporting / consensus) ══
    # ── QB ─────────────────────────────────────────────────────────────────
    "Lamar Jackson":         1.18,   # Model had him QB16 off a 16.8-PPG/13-game 2025; consensus QB2, healthy,
                                     #   confirmed starter. Was 1.25, which overshot its own target — it aimed at
                                     #   consensus, but ESPN has Allen 31st and Lamar 39th, and 1.25 put Lamar
                                     #   ahead of him. Trimmed to land just behind Allen, where consensus has him
    "Dak Prescott":          1.10,   # Model QB19 vs consensus QB8; healthy and confirmed
    "Fernando Mendoza":      0.35,   # NOT the LV starter — Kubiak named Cousins QB1; GM wants the rookie to sit year one
    # Kyler Murray is deliberately NOT listed. He carried a 0.80 for a "genuine open MIN
    # competition w/ J.J. McCarthy" — O'Connell named him the Week 1 starter on Aug 11, 2026
    # and McCarthy the backup, so that discount no longer has a reason to exist. His
    # availability risk is still priced in via the 14-game PROJ_GAMES_OVERRIDES entry.
    # Shough and Willis are deliberately NOT listed. Their old QB4/QB14 placements were
    # an artefact of force-includes being handed a flat 16 games; the availability fix in
    # _force_include_proj_games now sits them on the positional centre, which lands both
    # at consensus (~QB21/QB22) on its own. A multiplier here would double-count.
    # ── RB ─────────────────────────────────────────────────────────────────
    "Kenneth Walker III":    1.35,   # SB LX MVP, signed KC 3-yr/$45M as the lead back; consensus RB9-12 vs model RB~30
    # PIT is a 1-2 punch, not a workhorse. These were 1.15 / 0.72 on early-camp rep
    # reports; Pittsburgh's first preseason depth chart (Aug 5) lists WARREN as RB1
    # with Dowdle RB2, McCarthy has said he is planning a committee, and ESPN has
    # them one apart at RB28/RB29. The old pair had them 18 board spots apart.
    "Rico Dowdle":           0.95,
    "Jaylen Warren":         1.05,
    "Kenneth Gainwell":      0.80,   # TB pass-down back behind Irving. Was 0.55, written when ESPN had him ~RB101;
                                     #   the Aug-19 list has him RB31 and his 2025 (17 g, 13.0 PPG) was a real breakout
    "Javonte Williams":      0.80,   # Model RB6 vs ESPN 41 — largest unsupported RB spike on the board
    "D'Andre Swift":         0.88,   # Was 1.00, granted only because Monangai's hyperextended knee put his Wk1
                                     #   in doubt. That has expired — Monangai is back at RB2 and being drafted
                                     #   around RB34 as a sleeper — so the reason for the boost is gone. Ben
                                     #   Johnson ran a true split in DET (Gibbs/Montgomery) and reporting points
                                     #   to nearer 50-50 here, which the model's RB10 slot does not price. Not
                                     #   back to 0.85: Swift's 2025 (257 touches, 1,386 yards) was his best year
                                     #   and the market may be over-correcting. Sized between the two views
    "Ashton Jeanty":         1.15,   # Consensus RB6-7 vs model RB13; market is buying the Year-2 breakout
    # ── WR ─────────────────────────────────────────────────────────────────
    "Stefon Diggs":          0.85,   # WAS WR2 opposite McLaurin; real target volume, but Daniels is run-first and he has zero camp ramp
    "Terry McLaurin":        1.20,   # WAS alpha, healthy and signed long-term; consensus WR23 vs model WR~40. Trimmed slightly for Diggs
    "Jaylen Waddle":         1.25,   # Consensus WR16 in DEN alongside Nix; model had him WR~30
    "DJ Moore":              1.25,   # Consensus WR24 after the trade to BUF; model had him WR~45
    "Tetairoa McMillan":     1.12,   # Consensus WR13 vs model WR19
    "Jameson Williams":      0.80,   # DET WR2 behind St. Brown; model WR11 vs consensus WR26
    "Wan'Dale Robinson":     0.75,   # Slot-only, and TEN drafted Carnell Tate; model WR22 vs ESPN 106
    "Chris Olave":           0.92,   # NO camp reports flag ongoing availability concerns despite the extension
    "Davante Adams":         0.92,   # Age 33/34 behind Nacua in LAR; consensus WR28
    # ══ AUG-24-2026 AUDIT (vs the refreshed Aug-19 ESPN board) ═════════════════
    # ── RB ─────────────────────────────────────────────────────────────────
    "Derrick Henry":         1.20,   # Model 12.0 PPG vs an actual 16.4 over all 17 games in 2025 — the age
                                     #   curve is charging him for a decline his own rate already shows. ESPN RB10
    "TreVeyon Henderson":    0.88,   # Stevenson is trending as the NE Week 1 lead back and Henderson left the
                                     #   Aug-24 practice with a right leg/ankle issue (no diagnosis). Model RB19
                                     #   vs ESPN RB26 — RECHECK before drafting, this one is still unresolved
    "Mike Washington Jr.":   2.90,   # The largest override on the board, because the model has no way to see
                                     #   what makes him valuable. _rookie_base_ppr reads only draft capital, and
                                     #   pick 115 sits five slots past a tier edge (64.0 → 43.0), so he was
                                     #   projecting 25 pts — RB~110, behind three rookie RB3s. He is the LV RB2
                                     #   outright, with only a 34-year-old Mostert behind him, handcuffing a
                                     #   bell-cow whose own Week 1 is in doubt. Priced against the veterans in
                                     #   that exact role rather than against his draft slot: Ray Davis 78,
                                     #   Sean Tucker 90, Blake Corum 107. This lands him ~72, just under Davis
    # ── WR ─────────────────────────────────────────────────────────────────
    "Tank Dell":             0.65,   # Missed all of 2025 (multi-ligament knee). The 2023/24 blend has him at
                                     #   12.8 PPG and WR27; ESPN has him WR55 and HOU has not promised a role
    "Parker Washington":     1.00,   # Was 0.82 as the "JAX WR3". He posted 12.3 PPG over 14 games in 2025 and
                                     #   ESPN moved him to WR33 — the discount no longer has a reason to exist
    "Khalil Shakir":         1.15,   # BUF's target leader two years running (10.8 PPG in 2025); the team tier
                                     #   plus the Joe Brady HC penalty stacked him down to 8.6. ESPN WR43
    # ── TE ─────────────────────────────────────────────────────────────────
    "Pat Freiermuth":        1.00,   # Was 0.78 for the Darnell Washington extension, but that is already priced
                                     #   into his own falling rate. ESPN still has him TE22; model had him TE47
    "AJ Barner":             0.85,   # One 17-game season at 8.7 PPG; the age curve pushed him to 10.0 and TE12
                                     #   on a SEA offense that just drafted competition. ESPN TE26
    # ── TE ─────────────────────────────────────────────────────────────────
    "T.J. Hockenson":        1.20,   # Restructured to stay MIN and remains the lead TE; consensus TE19 vs model TE~35
    "Kenyon Sadiq":          1.30,   # NYJ traded up to take him 16th overall; immediate lead TE, consensus TE22
    "Colston Loveland":      1.15,   # Was CHI's de facto No. 1 target late in 2025 (10+ targets in each of the last four games)
    "Gunnar Helm":           1.10,   # TEN TE room is his alone now that Okonkwo left for WAS
    "Jadarian Price":        1.18,   # The rookie curve reads draft capital, and pick 32 lands on the wrong
                                     #   side of a tier edge, so a first-round back tagged "starter" in
                                     #   rookies_2026.csv still projected RB30 at NEGATIVE VOR. Walker left
                                     #   for KC and Charbonnet opened on PUP with a torn ACL, so the Week 1
                                     #   job is his against only Holani and Emanuel Wilson. Not a bell cow —
                                     #   SEA plan to use Wilson, and Charbonnet returns at some point — so
                                     #   this lands him a back-end RB2, near ESPN's 74th overall (SEA)
    # ══ AUG-30-2026: the GB backfield behind Jacobs ════════════════════════════
    # Both of these are career backups with too few games to escape the
    # REGRESS_MIN_GAMES shrinkage, so the engine hands them the positional median
    # instead of their own rate and they surfaced above Lloyd — and Johnson above
    # Jacobs. These pull them back to the roles beat reporting actually describes.
    "Kaleb Johnson":         0.66,   # 28 carries for 69 yards as a rookie, and PIT had him behind Lew Nichols
                                     #   and Travis Homer before trading him. Arrives days before Week 1 without
                                     #   the playbook — a distant second to Lloyd for the vacated work (GB)
    "Chris Brooks":          0.62,   # 2.71 PPR/g over 16 career games; a third-down and pass-protection back.
                                     #   Shrinkage had him at 7.36 PPG, which is a committee-lead rate (GB)
    # ══ AUG-30-2026: Josh Jacobs ═══════════════════════════════════════════════
    "Josh Jacobs":           0.92,   # Separate from the games cut, which only prices the weeks he misses.
                                     #   Exempt-list players may not practice at all, so he returns cold off a
                                     #   ~2-month layoff into a backfield someone else has been running. Rate,
                                     #   not just availability, takes the hit for the games he does play (GB)
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
    "Chris Godwin Jr.":       1996,
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
    "Brian Thomas Jr.":       2002,
    "Emeka Egbuka":           2002,
    "Tetairoa McMillan":      2003,
    "Marvin Harrison Jr.":    2003,
    "Malik Nabers":           2003,
    # ── Tight Ends ────────────────────────────────────────────────────────────
    "Travis Kelce":        1989,
    "Taysom Hill":         1990,
    "Darren Waller":       1992,
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
    "Chig Okonkwo":        2000,
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


# ── AGE-CURVE SOFTENING ───────────────────────────────────────────────────────
# _age_factor FORECASTS decline. But when a player's most recent season was
# already played inside the penalised band and he still beat his own prior
# baseline, that decline has been MEASURED, not predicted — and the projection
# already carries the result, because it is built from recency-weighted PPG.
# Multiplying the full penalty on top double-counts the same year of aging.
#
# Worked example (the case that surfaced this): Matthew Stafford, age 38 in 2026.
# His projection is 65% weighted to a 20.85-PPG age-37 season, the best fantasy
# year of his career. The model then applied 0.88x for "age 37+ cliff" — a cliff
# his own most recent tape shows no sign of. That single multiplier moved him
# from ~QB7 to QB17.
#
# So: forgive part of the penalty in proportion to how far the latest season ran
# ahead of the player's prior baseline. A player who is actually declining has a
# ratio <= 1.0 and keeps the full penalty.
AGE_SOFTEN_MAX_SHARE = 0.60   # never forgive more than 60% of the penalty
AGE_SOFTEN_FULL_AT   = 1.25   # latest PPG 25%+ over baseline earns the full share
AGE_SOFTEN_MIN_GAMES = 8      # ignore part-seasons: an injury year is not an age year


def _age_soften(mult: float, pos: str, age_2026: int, form_ratio: float | None) -> float:
    """Damp an age penalty the player's own recent production already disproves.

    Returns `mult` unchanged when there is nothing to forgive: no penalty, no
    comparable prior season, a latest season that did NOT beat the baseline, or
    a latest season played before the age curve started biting (in which case
    the penalty really is a forecast and should stand at full strength).
    """
    if mult >= 1.0 or form_ratio is None or form_ratio <= 1.0:
        return mult
    if _age_factor(pos, age_2026 - 1) >= 1.0:
        return mult   # last season predates the decline band — nothing measured yet
    share = min((form_ratio - 1.0) / (AGE_SOFTEN_FULL_AT - 1.0), 1.0) * AGE_SOFTEN_MAX_SHARE
    return mult + (1.0 - mult) * share


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

# 2026 projected-games overrides — curated availability for players whose
# HISTORICAL games misrepresent their 2026 outlook, so the data-driven expected-
# games model in model/projection.py can't infer it from prior seasons alone:
#   • suspensions (games lost for non-injury reasons)
#   • carry-over injuries that happened LATE (a mostly-full prior season hides a
#     recovery that bleeds into Week 1), or new offseason injuries
# Everyone NOT listed here keeps the model's own player-specific expected games.
# Predicted points are recomputed as pred_ppg × games, so the per-game rate is
# preserved and only availability changes. Format: name_fragment → 2026 games.
# Sources: 2026 offseason reporting (CBS/FantasyPros/NBC/4for4, Mar–Aug 2026).
PROJ_GAMES_OVERRIDES = {
    # 2025 suspension served; healthy for 2026 (ready for camp per Jul-2026 reporting),
    # projected as a full-season starter. Prior-year 8-game total understates his role.
    "Rashee Rice":      15,
    # Carry-over / offseason injuries the prior-season game count understates
    "Patrick Mahomes":  14,   # ACL recovery; Week 1 availability uncertain
    "Daniel Jones":     13,   # Torn Achilles Wk14 2025; 6–8mo recovery, Wk1 uncertain (IND)
    "Kyler Murray":     14,   # Foot cost ~11 games 2025; Wk1 starter but ~14% career miss rate (MIN)
    "Jayden Daniels":   14,   # 7 games 2025 (knee/hamstring/elbow); aggressive rushing profile (WAS)
    "Puka Nacua":       15,   # Elite when active but recurring lower-body history (2024 knee, 2025 ankle) (LAR)
    # 2026 top-150 audit — offseason injuries the prior game count understates
    "George Kittle":    13,   # Torn Achilles Jan-2026; opened camp Active/PUP, Reserve/PUP would cost 4 games (SF)
    "Tucker Kraft":     14,   # Off PUP but practicing limited (individual/walkthrough only); Wk1 is a target, not a lock (GB)
    "Chris Godwin Jr.": 13,   # Ankle PUP watch; slow start likely (TB)
    "Jordan Addison":   14,   # 3-game suspension to open 2026 (MIN)
    "Alec Pierce":      15,   # Was 12 on a behind-schedule PUP report; now expected off PUP within a week and available Wk1 (IND)
    "Zach Charbonnet":  8,    # ACL recovery; limited early-season availability (SEA)
    "Malik Nabers":     15,   # Was 12 on "no target date"; now taking full-speed 11-on-11 reps and on track for Wk1 (NYG)
    # ── Aug-2026 soft audit ────────────────────────────────────────────────
    "Stefon Diggs":     15,   # Signed with WAS Aug 5 with no camp ramp; slow September likely
    "Brock Bowers":     16,   # "100%" after the 2025 PCL/bone bruise; full speed through OTAs and minicamp (LV)
    "Cam Skattebo":     16,   # Fully cleared and a full participant; camp reports have him as the NYG RB1
    "Omarion Hampton":  16,   # Healthy and named the LAC "clear-cut featured back"
    "Quinshon Judkins": 15,   # Cleared to play all three downs, but held out of practice Aug 18-20 with a
                              #   "nagging" issue — precautionary per Monken, who still expects him Week 1 (CLE)
    # ── Aug-30-2026: Jacobs is no longer a "risk", he is already unavailable ───
    # Charged Aug 27 with misdemeanor battery and criminal damage to property over
    # the May 23 Hobart incident, and placed on the commissioner's exempt list on
    # Aug 30 — he cannot practice or attend games, so Week 1 is gone outright.
    # The exempt list runs until the case resolves, and the Personal Conduct
    # baseline for a domestic-violence finding is six games on top of it, with
    # prosecutors saying part of the incident is on video. Blending the outcomes
    # (quick plea and back by Wk4 / six-game ban with exempt time credited / exempt
    # list dragging into a full six-game ban / longer ban) lands near ten games.
    "Josh Jacobs":      10,
    # The GB backfield that inherits those weeks. Lloyd has managed one game in two
    # seasons (hamstring, then IR), so he gets the durability haircut even as the
    # projected starter; Johnson and Brooks are healthy.
    "MarShawn Lloyd":   14,
    # Johnson and Brooks are priced through games rather than rate, because rate is
    # not reachable for them: with 9 and 16 career games they sit under
    # REGRESS_MIN_GAMES, so _regress_ppg blends ~53% and ~38% of the RB starter
    # median (14.9 PPG) into players whose own rates are 1.0 and 2.7. Probing the
    # pipeline, Johnson's floor is 7.86 PPG even at a multiplier of zero. These are
    # therefore "games of usable role", not availability, and are the honest lever
    # until that shrinkage gets a role-aware exemption.
    "Kaleb Johnson":    11,   # third in line, no playbook, ~7 weeks of committee work then buried
    "Chris Brooks":     8,    # pass-protection specialist; real snaps, few touches
    # ── Aug-19-2026 preseason audit: new camp injuries ─────────────────────
    "Alvin Kamara":     11,   # Sprained MCL in the Aug-19 joint practice vs DAL; 4-6 wks, so ~3 games missed (NO)
    "Jordyn Tyson":     10,   # Hamstring, ~2 months (into mid-Oct) — prime candidate to open on IR, which forces 4+ games (NO)
    "Kyle Monangai":    12,   # Hyperextended right knee; "multiple weeks" and week-to-week, Wk1 in doubt (CHI)
    "Jeremiyah Love":   14,   # High ankle sprain (Aug 13), out for the rest of the preseason. ARI is "hopeful"
                              #   for Wk1 but follow-up reporting says it is worse than first portrayed (ARI)
    "TreVeyon Henderson": 14, # Left the Aug-24 practice with a right leg/ankle issue; no diagnosis yet (NE)
    "Ashton Jeanty":    15,   # Week 1 in doubt (Aug 2026). Pinned rather than left to the model, which
                              #   happens to land on 15 today but would drift on the next retrain (LV)
    # ── Aug-24-2026 audit: QBs the availability model double-counts ────────────
    # Their recency-weighted PPG already absorbs the injury seasons; letting the
    # durability model ALSO cut their games to ~11-12 charged them twice and left
    # three consensus top-12 QBs sitting at QB16-QB19 on rate alone.
    "Joe Burrow":       15,   # Turf-toe surgery (2025) fully healed; 17 games in 2024 (CIN)
    "Brock Purdy":      14,   # Toe/shoulder history is real, but he is the uncontested SF QB1
    # 13 games in 2025 was a Week-4 takeover, not injury, so the durability model was
    # reading a role artefact as fragility. Kept below a full season anyway: at 16 his
    # rate (3rd-best on the board off one partial year) vaulted him past Allen/Lamar.
    "Jaxson Dart":      14,
}

# ══════════════════════════════════════════════════════════════════════════════
# ROOKIES — 2026 draft class (draft-capital projection)
#
# Rookies have zero NFL history, so the ridge/PPG engine in model/projection.py
# cannot score them. Research (PlayerProfiler, Dynasty Nerds, PFF) is unanimous
# that DRAFT CAPITAL is the single most predictive input for rookie-season
# fantasy production, followed by landing spot (team offensive environment) and
# depth-chart role. We model each of those explicitly:
#   1. base points  = f(position, overall draft pick)   ← draft capital
#   2. × role mult   (depth-chart opportunity)            ← committee / backup
#   3. × scoring scalar (PPR / Half / Standard)
#   4. → then the SAME team-offense tier, new-HC, and age-curve overlays that
#        every veteran gets are applied downstream in apply_expert_adjustments,
#        which is where the "landing spot" signal enters (a WR on DET's #1
#        passing offense is boosted; an RB on a bottom rushing offense is cut).
# Rookie rows are injected into the board before those overlays so they flow
# through VOR and round-grade assignment exactly like any other player.
#
# Source: 2026 NFL Draft results (Rounds 1-6, skill positions) + post-draft
# dynasty consensus for depth-chart role. Rookie data lives in
# data/raw/rookies_2026.csv so the class can be edited without touching code.
# ══════════════════════════════════════════════════════════════════════════════

ROOKIE_CSV = RAW_DIR / "rookies_2026.csv"

# Kickers can't be projected by the statistical engine — nflverse weekly stats
# don't include FG/XP scoring, so kicker fantasy_points are ~0. Instead they're
# injected from a curated, editable projection file (points ≈ scoring-independent).
KICKER_CSV = RAW_DIR / "kickers_2026.csv"

# Team defenses (D/ST), same story as kickers: nflverse weekly stats are per
# player, so team-defense fantasy scoring lives in its own curated file.
DEFENSE_CSV = RAW_DIR / "defenses_2026.csv"

# ESPN 2026 positional rankings (Mike Clay PPR). Displayed on the big board next
# to the model's own overall rank so users can compare the model vs consensus.
# Matched to board players by normalized name so suffix / punctuation differences
# (Jr., III, "A.J." vs "AJ") don't break the join.
ESPN_RANKS_CSV = RAW_DIR / "espn_ranks_2026.csv"

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Nicknames ESPN uses that suffix-stripping alone can't reconcile with the
# nflverse spelling the board carries.
_NAME_ALIASES = {"kenny gainwell": "kenneth gainwell"}


def _norm_name(n: str) -> str:
    """Lowercase, strip punctuation and generational suffixes for name matching."""
    n = str(n).lower().replace(".", "").replace(",", "").strip()
    n = " ".join(p for p in n.split() if p not in _NAME_SUFFIXES)
    return _NAME_ALIASES.get(n, n)

# Depth-chart role → opportunity multiplier and projected games. QBs that sit
# behind a veteran ("redshirt") accrue almost no fantasy value, hence the steep
# cut; skill "backup" players still see rotational / injury-fill snaps.
#
# "handcuff" is the RB2 directly behind a bell-cow: he is active every week and
# is one injury from the lead job, but the starter's workload caps his weekly
# floor, so he earns neither the carry split a "committee" back gets nor the
# third-string treatment of a "backup". Added Aug-29-2026 because the taxonomy
# had no slot for it — Mike Washington Jr. is the only RB2 in the rookie class
# and was landing below three RB3s purely on the label.
ROOKIE_ROLE_MULT   = {"starter": 1.00, "committee": 0.78, "handcuff": 0.70, "backup": 0.45, "redshirt": 0.12}
ROOKIE_PROJ_GAMES  = {"starter": 16,   "committee": 15,   "handcuff": 15,   "backup": 14,   "redshirt": 6}


def _rookie_base_ppr(pos: str, pick: int) -> float:
    """Expected rookie-season PPR points as a function of overall draft pick.

    Tiers are grounded in historical rookie-year averages by draft slot: earlier
    capital → more guaranteed opportunity → more production. RB/WR curves are the
    steepest (early picks routinely start); rookie TEs historically produce little
    regardless of capital (Bowers-tier is the rare exception, not the baseline).

    Recalibrated in the Aug-24-2026 audit. The prior tops of the RB and WR curves
    were set below what first-round rookies actually deliver — Jeanty/Bijan/Gibbs
    all cleared 245 PPR, and Nabers/Egbuka/Harrison/McMillan put the top-10 rookie
    WR median near 190 — which pushed the entire 2026 rookie class 20-70 spots
    below consensus at every pick slot.
    """
    if pos == "RB":
        if pick <= 5:   return 225.0
        if pick <= 15:  return 178.0
        if pick <= 32:  return 145.0
        if pick <= 50:  return 115.0
        if pick <= 75:  return 88.0
        if pick <= 110: return 64.0
        if pick <= 150: return 43.0
        return 26.0
    if pos == "WR":
        if pick <= 8:   return 178.0
        if pick <= 20:  return 148.0
        if pick <= 32:  return 124.0
        if pick <= 50:  return 95.0
        if pick <= 75:  return 70.0
        if pick <= 110: return 50.0
        if pick <= 150: return 31.0
        return 16.0
    if pos == "TE":
        if pick <= 20:  return 105.0
        if pick <= 45:  return 70.0
        if pick <= 75:  return 50.0
        if pick <= 110: return 32.0
        return 16.0
    if pos == "QB":
        # QB points are ~scoring-format-independent (QBs rarely catch passes).
        # A benched rookie ("redshirt") is scaled to near-zero by ROOKIE_ROLE_MULT.
        if pick <= 5:   return 260.0
        if pick <= 15:  return 235.0
        if pick <= 40:  return 205.0
        return 180.0
    return 0.0


def _rookie_scoring_scalar(pos: str, scoring: str) -> float:
    """Scale reception-driven rookie points for non-PPR formats (QBs unaffected)."""
    if pos == "QB":
        return 1.0
    table = {
        "PPR":      {"RB": 1.00, "WR": 1.00, "TE": 1.00},
        "Half PPR": {"RB": 0.93, "WR": 0.90, "TE": 0.88},
        "Standard": {"RB": 0.86, "WR": 0.78, "TE": 0.75},
    }
    return table.get(scoring, table["PPR"]).get(pos, 1.0)


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

# Scoring format selector — drives TARGET_COL and REPLACEMENT_LEVEL below.
# Half PPR is computed as (Standard + PPR) / 2  (mathematically exact:
# half-PPR awards 0.5 pts/reception, which is the midpoint of 0 and 1).
sel_scoring = st.sidebar.radio(
    "Scoring Format",
    ["PPR", "Half PPR", "Standard"],
    key=f"pred_scoring_{_v}",
    help="Switches the projection target and recalibrates VOR replacement levels."
)

sel_pos = st.sidebar.selectbox("Position", ["All"] + list(POSITION_FEATURES.keys()) + ["K", "DEF"],
                               key=f"pred_pos_{_v}")
top_n = st.sidebar.slider("Big Board Size", 10, 200, 100, key=f"pred_top_{_v}")

if st.sidebar.button("Reset Filters", key="pred_reset", use_container_width=True):
    st.session_state["pred_v"] = _v + 1
    st.rerun()

# ── Apply scoring format ─────────────────────────────────────────────────────
# Derive the Half-PPR column on the fly if needed. Mutates the in-memory
# weekly DataFrame for this run only (the cached underlying df is not modified
# because load_weekly returns a fresh reference each Streamlit run).
if sel_scoring == "Half PPR" and "fantasy_points_half_ppr" not in weekly.columns:
    if {"fantasy_points", "fantasy_points_ppr"}.issubset(weekly.columns):
        weekly = weekly.copy()
        weekly["fantasy_points_half_ppr"] = (
            (weekly["fantasy_points"] + weekly["fantasy_points_ppr"]) / 2.0
        ).round(2)

# Reassign module-level constants based on scoring selection.
# These globals are read by build_predictions() and apply_expert_adjustments().
TARGET_COL = SCORING_TARGET_COLS.get(sel_scoring, "fantasy_points_ppr")
if TARGET_COL not in weekly.columns:
    st.warning(f"Column {TARGET_COL!r} not found — falling back to PPR.")
    TARGET_COL = "fantasy_points_ppr"

REPLACEMENT_LEVEL = SCORING_REPLACEMENT_LEVELS.get(
    sel_scoring, SCORING_REPLACEMENT_LEVELS["PPR"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building 2026 projections …")
def build_predictions(weekly_df: pd.DataFrame, target_col: str = "fantasy_points_ppr"):
    # target_col is captured as a cache key so projections rebuild when scoring
    # format changes (PPR / Half PPR / Standard). The function body still reads
    # the module-level TARGET_COL global so the rest of the file stays simple.
    global TARGET_COL
    TARGET_COL = target_col

    # Pure ridge/PPG-blend engine now lives in model/projection.py — no globals,
    # explicit signature, reused as-is by scripts/backtest_model.py. This call
    # (as_of_season=None) trains on every season pair in weekly_df, same as the
    # original inline implementation.
    config = ProjectionConfig(
        target_col=TARGET_COL, name_col=name_col, pos_col=pos_col,
        team_col=team_col, track_col=track_col,
    )
    all_preds, hist = build_predictions_core(weekly_df, config)
    if all_preds.empty:
        return all_preds, hist

    # ── Injury risk flag ──────────────────────────────────────────────────────
    # build_predictions_core already sets the QB auto-flag (avg games/yr < 14.5
    # over last 3 seasons). Overlay the hand-curated INJURY_RISK_MAP (expert
    # research, March 2026) on top for ALL positions — RB/WR/TE have no
    # auto-flag, so this reduces to a pure map lookup for them, same as before.
    all_preds["injury_risk"] = all_preds.apply(
        lambda r: "      Yes      " if (
            r["injury_risk"] == "      Yes      "
            or INJURY_RISK_MAP.get(str(r[name_col]), "") != ""
        ) else "",
        axis=1
    )
    return all_preds, hist


all_preds_raw, hist_totals = build_predictions(weekly, target_col=TARGET_COL)


@st.cache_data(show_spinner=False)
def load_rookies() -> pd.DataFrame:
    """Load the 2026 rookie draft class from data/raw/rookies_2026.csv."""
    if not ROOKIE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ROOKIE_CSV)


# Interleave ESPN's positional ranks into a single overall draft order. ESPN's
# published overall cheat sheet is essentially this same interleave by draft
# value; these per-position value curves reproduce a standard 1-QB PPR cadence
# (RB/WR run together at the top, first TE lands early Rd 2, first QB early Rd 3).
_ESPN_VALUE_BASE  = {"RB": 100.0, "WR": 100.0, "TE": 92.5, "QB": 85.0}
_ESPN_VALUE_SLOPE = {"RB": 1.30,  "WR": 1.15,  "TE": 2.50, "QB": 1.60}
_ESPN_POS_TIEBREAK = {"RB": 0, "WR": 1, "TE": 2, "QB": 3}


@st.cache_data(show_spinner=False)
def load_espn_ranks() -> dict:
    """Map normalized player name → ESPN overall rank (int as string).

    Reads positional ranks (e.g. 'RB6') from data/raw/espn_ranks_2026.csv and
    collapses them into a single cross-position overall ordering by draft value.
    """
    if not ESPN_RANKS_CSV.exists():
        return {}
    df = pd.read_csv(ESPN_RANKS_CSV)

    scored = []
    for _, r in df.iterrows():
        label = str(r["espn"]).strip().upper()
        pos = "".join(c for c in label if c.isalpha())
        digits = "".join(c for c in label if c.isdigit())
        if pos not in _ESPN_VALUE_BASE or not digits:
            continue
        pos_rank = int(digits)
        value = _ESPN_VALUE_BASE[pos] - _ESPN_VALUE_SLOPE[pos] * (pos_rank - 1)
        scored.append((value, _ESPN_POS_TIEBREAK[pos], pos_rank, _norm_name(r["player"])))

    # Higher value first; ties broken by position priority then positional rank.
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return {name: i for i, (_, _, _, name) in enumerate(scored, start=1)}


@st.cache_data(show_spinner=False)
def build_rookie_predictions(target_col: str, scoring: str) -> pd.DataFrame:
    """Draft-capital-based projections for the 2026 rookie class.

    Returns rows in the same shape build_predictions_core emits so they concat
    cleanly onto the veteran board and flow through every downstream overlay
    (team tier, new-HC penalty, age curve, VOR, round grade).
    """
    rk = load_rookies()
    if rk.empty:
        return pd.DataFrame()

    rows = []
    for _, r in rk.iterrows():
        pos = str(r["position"]).upper()
        if pos not in POSITION_FEATURES:
            continue
        role = str(r["role"]).lower()
        base = _rookie_base_ppr(pos, int(r["pick"]))
        base *= ROOKIE_ROLE_MULT.get(role, 0.45)
        base *= _rookie_scoring_scalar(pos, scoring)
        proj_g = ROOKIE_PROJ_GAMES.get(role, 14)
        pred   = round(base, 1)

        row = {
            name_col:        r["player"],
            pos_col:         pos,
            "season":        PREDICTION_YEAR - 1,
            "games":         0,                       # no prior NFL season
            target_col:      0.0,                     # no last-season points
            "predicted_pts": pred,                    # pre-overlay base (tier/HC/age applied later)
            "proj_games":    float(proj_g),
            "pred_ppg":      round(pred / max(proj_g, 1), 2),
            "rmse":          0.0,
            "injury_risk":   "",
            "is_rookie":     True,
        }
        if track_col != name_col:
            row[track_col] = f"ROOKIE-{r['player']}"
        if team_col:
            row[team_col] = r["team"]
        rows.append(row)

    return pd.DataFrame(rows)


# Inject rookies BEFORE expert overlays so team-tier / HC / age adjustments and
# VOR / round grades treat them exactly like veterans. Register their birth years
# first so _age_factor uses the real rookie age (≈22) instead of the veteran
# first-season fallback.
rookie_preds = build_rookie_predictions(TARGET_COL, sel_scoring)
if not rookie_preds.empty:
    _rk_meta = load_rookies()
    for _, _r in _rk_meta.iterrows():
        PLAYER_BIRTH_YEARS.setdefault(str(_r["player"]), int(_r["birth_year"]))
    all_preds_raw = pd.concat([all_preds_raw, rookie_preds], ignore_index=True)


def _force_include_proj_games(pos: str, player_name: str, reg_w: pd.DataFrame,
                               use_durability: bool = True) -> float:
    """Expected 2026 games for a force-included starter.

    Force-included players are here precisely BECAUSE their recent seasons were
    injury-shortened, so handing them a flat full-season count inverted the board:
    fragile players (Shough, Willis) outranked durable ones (Allen, Lamar) purely on
    an assumed availability edge they hadn't earned. Instead run them through the same
    availability model as everyone else — recency-weighted durability regressed toward
    the position's starter-cohort mean, centred on PPG_BASELINE_GAMES.

    Pass use_durability=False for career backups turned starter. Their thin game logs
    reflect ROLE, not fragility, so reading them as a durability signal would penalise
    a player for having been a backup; the positional centre is the honest answer.
    """
    base_g = float(PPG_BASELINE_GAMES.get(pos, MAX_PROJ_GAMES))
    pos_rows = reg_w[reg_w[pos_col] == pos]
    if pos_rows.empty or not use_durability:
        return base_g

    games = pos_rows.groupby([name_col, "season"])[TARGET_COL].count()
    by_player: dict[str, dict[int, float]] = {}
    for (nm, szn), g in games.items():
        by_player.setdefault(nm, {})[int(szn)] = float(g)
    durab = {nm: _weighted_durability(h, AVAIL_RECENCY_WEIGHTS) for nm, h in by_player.items()}

    # Centre on the same cohort build_predictions_core uses: players who posted a
    # qualifying (starter-level) season in the most recent year of data. Averaging over
    # every player at the position would drag the mean down with backups and inflate
    # everyone measured against it.
    min_g   = MIN_GAMES_BY_POS.get(pos, 6)
    latest  = int(max(s for h in by_player.values() for s in h))
    cohort  = [nm for nm, h in by_player.items() if h.get(latest, 0) >= min_g]
    vals    = [durab[nm] for nm in cohort if np.isfinite(durab.get(nm, np.nan))]
    pop_mean = float(np.mean(vals)) if vals else base_g

    return round(_expected_games(durab.get(player_name, float("nan")), pop_mean, base_g,
                                 AVAIL_SHRINK, AVAIL_GAMES_FLOOR, AVAIL_GAMES_CEILING), 1)


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
                # Career backup: the thin game log is a role artefact, not fragility,
                # so sit them on the positional centre rather than either assuming a
                # full season or punishing them for the backup years.
                proj_g   = _force_include_proj_games(pos, player_name, reg_w,
                                                     use_durability=False)
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
                # Same availability model as the rest of the board — their recent
                # missed time is real injury signal and should count against them.
                # Players with a specific 2026 situation (e.g. Kyler Murray, Jayden
                # Daniels) are still refined below via PROJ_GAMES_OVERRIDES.
                proj_g    = _force_include_proj_games(pos, player_name, reg_w)
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
                # Expert-supplied PPG (manual_ppg) is already the projection; the
                # small-sample regression / peak-cap below must NOT drag it toward
                # a generic positional median (that inflated e.g. Shough's 15.8 to
                # ~17.9 by blending halfway to the QB starter median).
                "_curated_ppg": manual_ppg is not None,
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

    # 4a. Team from the 2026 depth chart — the authoritative roster source.
    #     Without this the team column carries over from the last season a player
    #     actually logged snaps in, so anyone who moved in the offseason keeps his
    #     old club and then gets scored against that club's offensive tier. The
    #     hand-maintained dict below used to be the only correction, which meant a
    #     move was wrong until somebody noticed it; the depth chart already knows.
    #     Only offensive skill players are mapped — K and DEF are keyed differently.
    if team_col:
        _dc_path = RAW_DIR / "depth_charts.csv"
        dc = load_depth_charts(_mtime=_dc_path.stat().st_mtime if _dc_path.exists() else 0.0)
        if not dc.empty and {"season", "side", "position", "player_name", "team"} <= set(dc.columns):
            dc = dc[(dc["season"] == PREDICTION_YEAR) & (dc["side"] == "offense")
                    & (dc["position"].isin(["QB", "RB", "WR", "TE"]))]
            # Names are unique across the 2026 offensive depth charts, so a plain
            # name → team map is unambiguous; normalise so "Etienne Jr." matches
            # "Etienne" and "A.J." matches "AJ".
            dc_team = dict(zip(dc["player_name"].map(_norm_name), dc["team"]))
            mapped = out[name_col].map(_norm_name).map(dc_team)
            out.loc[mapped.notna(), team_col] = mapped[mapped.notna()]

    # 4b. Team corrections (trades / FA signings not captured in historical data).
    #     Applied last so a hand entry still wins: it covers players missing from
    #     the depth chart entirely and moves that land after the last chart pull.
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
        # Latest-season PPG divided by the mean of the two seasons before it.
        # Feeds _age_soften; absent when there's no comparable prior season.
        _form_ratio: dict = {}
        if raw_weekly is not None and not raw_weekly.empty:
            _raw_reg = raw_weekly
            if "season_type" in raw_weekly.columns:
                _raw_reg = raw_weekly[raw_weekly["season_type"] == "REG"]
            _first_szn = _raw_reg.groupby(name_col)["season"].min().to_dict()

            _szn_ppg = _raw_reg.groupby([name_col, "season"])[TARGET_COL].agg(["sum", "count"])
            _szn_ppg = _szn_ppg[_szn_ppg["count"] >= AGE_SOFTEN_MIN_GAMES]
            _szn_ppg["ppg"] = _szn_ppg["sum"] / _szn_ppg["count"]
            for _nm, _rows in _szn_ppg.groupby(level=0):
                _s = _rows.reset_index().sort_values("season", ascending=False)
                # Require the most recent season to actually be last year — a
                # player who sat out 2025 has no new evidence to offer.
                if len(_s) < 2 or _s.iloc[0]["season"] != PREDICTION_YEAR - 1:
                    continue
                _base = _s.iloc[1:3]["ppg"].mean()
                if _base > 0:
                    _form_ratio[_nm] = float(_s.iloc[0]["ppg"] / _base)

        def _player_age_2026(name: str, pos: str) -> int:
            if name in PLAYER_BIRTH_YEARS:
                return 2026 - PLAYER_BIRTH_YEARS[name]
            first = _first_szn.get(name, 2020)
            return 2026 - (first - _AVG_DRAFT_AGE.get(pos, 22))

        def _age_mult(r) -> float:
            name = str(r.get(name_col, ""))
            pos  = str(r.get(pos_col, ""))
            age  = _player_age_2026(name, pos)
            return _age_soften(_age_factor(pos, age), pos, age, _form_ratio.get(name))

        age_mults = out.apply(_age_mult, axis=1)
        out["predicted_pts"] = (out["predicted_pts"] * age_mults).round(1)
        out["pred_ppg"]      = (out["pred_ppg"]      * age_mults).round(2)

    # 8. Player-specific multipliers — applied last so they sit on top of every
    #    other contextual adjustment (team tier, HC, age). Keeps the methodology
    #    caption (Gibbs 1.22×, JSN 1.18×, Kelce 0.82×, etc.) honest by actually
    #    APPLYING the values it advertises.
    if name_col and PLAYER_MULTIPLIERS:
        player_mults = out[name_col].map(
            lambda n: PLAYER_MULTIPLIERS.get(str(n), 1.0)
        ).astype(float)
        out["predicted_pts"] = (out["predicted_pts"] * player_mults).round(1)
        out["pred_ppg"]      = (out["pred_ppg"]      * player_mults).round(2)

    # 9. Sanity guards on the final per-game rate — applied last, over each
    #    player's real game history: cap at their own ceiling, then regress
    #    tiny samples toward the position's established median. See constants.
    if "_curated_ppg" not in out.columns:
        out["_curated_ppg"] = False
    out["_curated_ppg"] = out["_curated_ppg"].fillna(False).astype(bool)

    if (name_col and raw_weekly is not None and not raw_weekly.empty
            and "season" in raw_weekly.columns):
        _reg = raw_weekly
        if "season_type" in raw_weekly.columns:
            _reg = raw_weekly[raw_weekly["season_type"] == "REG"]
        _tc = TARGET_COL if TARGET_COL in _reg.columns else "fantasy_points_ppr"
        _szn = (_reg.groupby([name_col, "season"])[_tc]
                    .agg(g="size", pts="sum").reset_index())
        _szn["ppg"] = _szn["pts"] / _szn["g"].clip(lower=1)
        career_g = _szn.groupby(name_col)["g"].sum().to_dict()
        peak_ppg = (_szn[_szn["g"] >= PEAK_MIN_GAMES]
                    .groupby(name_col)["ppg"].max().to_dict())

        # (a) Cap each player with real history at 105% of their best season.
        #     Deliberately applied AFTER the multipliers: a hand-entered boost is a
        #     hypothesis about role, but year-over-year regression toward a player's
        #     own demonstrated level is the stronger force, so the peak wins the tie.
        #     A multiplier pinned at this cap is a signal the boost is too aggressive.
        def _cap_ppg(row):
            ppg = float(row["pred_ppg"])
            if row["_curated_ppg"] or str(row[name_col]) in PEAK_CAP_EXEMPT:
                return ppg  # expert-supplied rate, or a promotion the peak can't see
            pk = peak_ppg.get(str(row[name_col]))
            return min(ppg, pk * PEAK_CAP_MULT) if pk is not None else ppg
        out["pred_ppg"] = out.apply(_cap_ppg, axis=1).round(2)

        # (b) Regress low-sample players toward a typical established starter at
        #     their position: the median PPG of players with enough career games
        #     who project above replacement level (excludes the deep-bench tail).
        if pos_col:
            _cg = out[name_col].map(lambda n: career_g.get(str(n), 0))
            _est = out[_cg >= REGRESS_MIN_GAMES]
            base = {}
            for _pos, _grp in _est.groupby(pos_col):
                _thr = REPLACEMENT_LEVEL.get(_pos, 0) / MAX_PROJ_GAMES
                _starters = _grp.loc[_grp["pred_ppg"] >= _thr, "pred_ppg"]
                base[_pos] = float(_starters.median() if not _starters.empty
                                   else _grp["pred_ppg"].median())

            def _regress_ppg(row):
                cg = career_g.get(str(row[name_col]), 0)
                ppg = float(row["pred_ppg"])
                if row["_curated_ppg"] or cg <= 0 or cg >= REGRESS_MIN_GAMES:
                    return ppg  # expert-curated, no history (rookies), or established
                b = base.get(row.get(pos_col, ""), ppg)
                w = cg / (cg + REGRESS_K)
                return w * ppg + (1 - w) * b
            out["pred_ppg"] = out.apply(_regress_ppg, axis=1).round(2)

        # Keep season totals consistent with the adjusted per-game rate.
        if "proj_games" in out.columns:
            out["predicted_pts"] = (out["pred_ppg"] * out["proj_games"]).round(1)

    return out.drop(columns=["_curated_ppg"], errors="ignore").reset_index(drop=True)


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


@st.cache_data(show_spinner=False)
def build_kicker_predictions() -> pd.DataFrame:
    """Curated 2026 kicker projections from data/raw/kickers_2026.csv.

    Kicker scoring is essentially format-independent, so a single projection is
    used for PPR/Half/Standard. Rows match the board schema so they flow through
    VOR and round-grade assignment exactly like every other player. They are
    injected AFTER the expert/age/games overlays (none of which apply to kickers).
    """
    if not KICKER_CSV.exists():
        return pd.DataFrame()
    kk = pd.read_csv(KICKER_CSV)
    if kk.empty:
        return pd.DataFrame()

    rows = []
    for _, r in kk.iterrows():
        pred = round(float(r["proj_pts"]), 1)
        row = {
            name_col:        str(r["player"]),
            pos_col:         "K",
            "season":        PREDICTION_YEAR - 1,
            "games":         0,
            TARGET_COL:      0.0,
            "predicted_pts": pred,
            "proj_games":    float(MAX_PROJ_GAMES),
            "pred_ppg":      round(pred / MAX_PROJ_GAMES, 2),
            "rmse":          0.0,
            "injury_risk":   "",
            "is_rookie":     False,
        }
        if track_col != name_col:
            row[track_col] = f"K-{r['player']}"
        if team_col:
            row[team_col] = str(r["team"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_defense_predictions() -> pd.DataFrame:
    """Curated 2026 team-defense (D/ST) projections from data/raw/defenses_2026.csv.

    Same shape and lifecycle as build_kicker_predictions: D/ST scoring is
    format-independent, so a single projection serves PPR/Half/Standard, and the
    rows flow through VOR and round-grade exactly like every other player.
    """
    if not DEFENSE_CSV.exists():
        return pd.DataFrame()
    dd = pd.read_csv(DEFENSE_CSV)
    if dd.empty:
        return pd.DataFrame()

    rows = []
    for _, r in dd.iterrows():
        pred = round(float(r["proj_pts"]), 1)
        row = {
            name_col:        str(r["player"]),
            pos_col:         "DEF",
            "season":        PREDICTION_YEAR - 1,
            "games":         0,
            TARGET_COL:      0.0,
            "predicted_pts": pred,
            "proj_games":    float(MAX_PROJ_GAMES),
            "pred_ppg":      round(pred / MAX_PROJ_GAMES, 2),
            "rmse":          0.0,
            "injury_risk":   "",
            "is_rookie":     False,
        }
        if track_col != name_col:
            row[track_col] = f"DEF-{r['team']}"
        if team_col:
            row[team_col] = str(r["team"])
        rows.append(row)
    return pd.DataFrame(rows)


# Inject kickers just before VOR so they rank by the same scarcity logic. Their
# replacement level (REPLACEMENT_LEVEL["K"]) is inflated above every kicker
# projection, so kicker VOR is strongly negative and they land in the final rounds.
_kicker_preds = build_kicker_predictions()
if not _kicker_preds.empty:
    all_preds = pd.concat([all_preds, _kicker_preds], ignore_index=True)

# Team defenses, same treatment (inflated REPLACEMENT_LEVEL["DEF"] sorts them last).
_defense_preds = build_defense_predictions()
if not _defense_preds.empty:
    all_preds = pd.concat([all_preds, _defense_preds], ignore_index=True)


def _assign_vor(df: pd.DataFrame) -> pd.DataFrame:
    """Add VOR and model-derived round grade columns.

    VOR = predicted_pts − replacement_level[position], where replacement level is
    derived from THIS board's own projections (see below). Sorting by VOR rather
    than raw points accounts for positional scarcity — an elite TE ranks higher
    than an equivalent-points RB.

    round_grade is derived entirely from this model's own VOR rankings:
      Overall model rank 1–10   → Rd 1
      Overall model rank 11–20  → Rd 2
      … and so on (LEAGUE_SIZE picks per round).
    No external ADP data is used — the grade reflects what our projections say,
    not what consensus thinks.
    """
    out = df.copy()

    # ── Replacement level, derived from the projection pool ──────────────────
    # Previously these were hardcoded constants read off ACTUAL season finishes
    # (~16.5 real games played). But projections sit on a ~13-game baseline, so
    # the two sides of the subtraction were denominated in different units and
    # every QB came out roughly 50 points too negative — only one QB on the whole
    # board cleared a 290 baseline. Deriving the baseline from the same pool the
    # players are scored from puts both sides in one currency, and makes the
    # result follow ROSTER_SLOTS instead of a frozen 2024-25 snapshot.
    _roster = out[out[pos_col].isin(ROSTER_SLOTS)]
    _derived, _ = derive_replacement_baseline(
        _roster, pos_col, "predicted_pts", ROSTER_SLOTS, FLEX_ELIGIBLE,
        LEAGUE_SIZE, name_col=name_col,
    )
    # K and DEF are deliberately left alone. Their entries aren't real replacement
    # levels — they're sentinels inflated above every projection so kickers and
    # defenses sort into the final rounds, mirroring how real drafts treat them.
    REPLACEMENT_LEVEL.update({p: round(v, 1) for p, v in _derived.items()})

    # ── VOR ──────────────────────────────────────────────────────────────────
    out["vor"] = out.apply(
        lambda r: round(float(r["predicted_pts"]) - REPLACEMENT_LEVEL.get(r[pos_col], 0), 1),
        axis=1,
    )

    # ── Model rank → round grade ──────────────────────────────────────────────
    # Sort all players together by VOR so round grades are cross-positional
    # (e.g. an elite TE earns a Rd 1 grade if the model projects him top-10).
    out = out.sort_values("vor", ascending=False).reset_index(drop=True)
    model_ranks = range(1, len(out) + 1)
    out["round_grade"] = [
        f"Rd {int((rank - 1) // LEAGUE_SIZE) + 1}" for rank in model_ranks
    ]

    return out


all_preds = _assign_vor(all_preds)


# ── Persist finished board for the Draft Simulator ───────────────────────────
BIG_BOARD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "derived"


def _persist_big_board(board: pd.DataFrame, scoring: str) -> None:
    """Write the finished VOR-sorted board to parquet so the Draft Simulator
    consumes the exact same rankings (incl. ESPN overall rank) shown here."""
    if board.empty:
        return
    espn_map = load_espn_ranks()
    n = len(board)
    out = pd.DataFrame({
        "player":        board[name_col].astype(str).values,
        "pos":           board[pos_col].astype(str).values,
        "team":          board[team_col].astype(str).values if team_col else [""] * n,
        "predicted_pts": board["predicted_pts"].values,
        "pred_ppg":      board["pred_ppg"].values,
        "proj_games":    board["proj_games"].values,
        "vor":           board["vor"].values,
        "round_grade":   board["round_grade"].values,
        "injury_risk":   board["injury_risk"].values if "injury_risk" in board.columns else [""] * n,
        "is_rookie":     board["is_rookie"].fillna(False).values if "is_rookie" in board.columns else [False] * n,
    })
    out["espn_overall"] = out["player"].map(lambda nm: espn_map.get(_norm_name(nm)))
    try:
        BIG_BOARD_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(BIG_BOARD_DIR / f"big_board_{scoring.replace(' ', '_')}.parquet", index=False)
    except Exception:
        pass


_persist_big_board(all_preds, sel_scoring)

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

# Rookie indicator for the board ("R" for 2026 draft class, blank otherwise).
_rookie_col = preds["is_rookie"] if "is_rookie" in preds.columns else pd.Series(False, index=preds.index)
preds["rookie_tag"] = _rookie_col.fillna(False).map(lambda x: "R" if bool(x) else "")

# ESPN rank (consensus reference shown beside the model's own rank).
# Look up each player's global ESPN overall value, then re-rank WITHIN the
# current (position-filtered, top-N) view so the column renumbers 1..N by ESPN
# order — exactly like the model's own Rank column re-bases when a position is
# selected. Players ESPN never ranked stay NA here and are parked at
# UNRANKED_SORT_VALUE at render time. Nullable Int64 keeps the sort numeric
# rather than lexicographic.
_espn_map = load_espn_ranks()
_espn_overall = pd.to_numeric(
    preds[name_col].map(lambda n: _espn_map.get(_norm_name(n))), errors="coerce"
)
preds["espn_rank"] = _espn_overall.rank(method="first", ascending=True).astype("Int64")

# 2025 final finish (actual prior-season result shown beside the projections).
# Re-ranked WITHIN the current position-filtered, top-N view by 2025 total
# fantasy points — exactly like the ESPN and model Rank columns — so it
# renumbers 1..N. Rookies and players with no 2025 production stay NA here and
# are parked at UNRANKED_SORT_VALUE at render time. Nullable Int64 keeps the
# sort numeric.
_last_pts = pd.to_numeric(preds["last_season_pts"], errors="coerce")
_no_2025 = _rookie_col.fillna(False).to_numpy() | (_last_pts <= 0)
preds["finish_2025"] = _last_pts.mask(_no_2025).rank(method="first", ascending=False).astype("Int64")

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

board_cols = ["Rank", "espn_rank", "finish_2025", name_col, "rookie_tag"]
if team_col: board_cols.append(team_col)
if pos_col:  board_cols.append(pos_col)
board_cols += ["injury_risk", "predicted_pts", "vor", "round_grade", "pred_ppg", "proj_games", "last_season_pts", "change", "change_pct", "games", "last_season_ppg"]

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
    "espn_rank": "ESPN",
    "finish_2025": "2025 Finish",
    "rookie_tag": "Rk",
    "injury_risk": "Injury Risk",
    "predicted_pts": "2026 Proj",
    "vor":          "VOR",
    "round_grade":  "Round",
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
# Clicking "ESPN" or "2025 Finish" ascending used to put the unranked players
# on top of rank 1 — see rank_display for why these render as padded text.
for _c in ("espn_rank", "finish_2025"):
    if _c in disp.columns:
        disp[_c] = rank_display(disp[_c])
for c in disp.select_dtypes("float").columns:
    if c != "change_pct":
        disp[c] = disp[c].round(1)

# Add logo URLs for team column if it exists
teams_df = load_teams()
column_config_dict = {
    "ESPN": st.column_config.TextColumn(
                      label="ESPN", width="small",
                      help="ESPN's 2026 PPR draft rank (Mike Clay), re-based to the current "
                           "position filter so it renumbers 1..N alongside this model's own Rank. "
                           f"{UNRANKED_MARK} = outside ESPN's ranked pool (sorts to the bottom)."),
    "2025 Finish": st.column_config.TextColumn(
                      label="2025 Finish", width="small",
                      help="Where the player actually finished in 2025 fantasy scoring "
                           "(this scoring format), re-based to the current position filter so it "
                           "renumbers 1..N alongside ESPN and this model's Rank. "
                           f"{UNRANKED_MARK} = no 2025 production (rookies / non-qualifiers), "
                           "sorts to the bottom."),
    "Rk": st.column_config.TextColumn(
                      label="Rk", width="small",
                      help="R = 2026 rookie. Rookies have no NFL history, so they are "
                           "projected from draft capital (overall pick) × depth-chart role, "
                           "then run through the same team-offense, new-HC, and age-curve "
                           "overlays as veterans."),
    "Injury Risk": st.column_config.TextColumn(
                      label="Injury Risk",
                      width="small",
                      help="Yes = Player has a documented injury history or is coming off a significant injury. "
                           "QBs: flagged if avg games/yr < 14.5 over last 3 seasons OR listed in expert injury map. "
                           "RB/WR/TE: flagged via expert research (significant injury in 2024-25 or chronic history). "
                           "Projections assume full 17-game season regardless of flag."),
    "2026 Proj":  st.column_config.NumberColumn(format="%.1f"),
    "VOR":        st.column_config.NumberColumn(format="%.1f",
                      help="Value Over Replacement — positional scarcity-adjusted score. "
                           "Accounts for how scarce elite players are at each position "
                           f"(QB={REPLACEMENT_LEVEL['QB']}, RB={REPLACEMENT_LEVEL['RB']}, "
                           f"WR={REPLACEMENT_LEVEL['WR']}, TE={REPLACEMENT_LEVEL['TE']} replacement baselines, "
                           "derived from this board's own projections and your roster slots)."),
    "Round":      st.column_config.TextColumn(
                      help=f"Suggested fantasy draft round derived from this model's own VOR rankings. "
                           f"Overall VOR rank 1–{LEAGUE_SIZE} = Rd 1, "
                           f"{LEAGUE_SIZE+1}–{LEAGUE_SIZE*2} = Rd 2, etc. "
                           f"({LEAGUE_SIZE}-team league, {LEAGUE_SIZE} picks per round). "
                           "Grades update automatically as projections change — no external ADP used."),
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
    f"**Scoring** — Active format: **{sel_scoring}** (target column: `{TARGET_COL}`). "
    "Use the sidebar to switch between PPR, Half PPR, and Standard — replacement levels and projections recalibrate. "
    "**Methodology** — Position-specific ridge regression trained on 2016–2025 consecutive-season pairs "
    "with exponential recency weighting (recent seasons count more). Features are per-game rates, not "
    "season totals, so a player who missed games due to injury is not penalised for low counting stats. "
    "**Availability model** — projected 2026 games are player-specific: a recency-weighted average of each "
    "player's recent-season games, regressed toward the positional average (durable players project for more "
    "games, injury-prone players fewer). This is centred on the position mean so it redistributes value without "
    "inflating the overall scale — points = projected per-game rate × expected games. QB qualifier: "
    f"{MIN_GAMES_BY_POS['QB']}+ games started. Skill positions: 6+ games. "
    f"**VOR (Value Over Replacement)** ranks players by positional scarcity in an {LEAGUE_SIZE}-team league: "
    f"elite TEs rank higher than equivalent-point WRs because only {LEAGUE_SIZE} starting TEs exist. Replacement levels "
    f"(QB={REPLACEMENT_LEVEL['QB']}, RB={REPLACEMENT_LEVEL['RB']}, WR={REPLACEMENT_LEVEL['WR']}, "
    f"TE={REPLACEMENT_LEVEL['TE']}) are derived from this board's own projections rather than hardcoded: "
    f"starting slots ({LEAGUE_SIZE}-team, "
    f"{ROSTER_SLOTS['QB']}QB/{ROSTER_SLOTS['RB']}RB/{ROSTER_SLOTS['WR']}WR/{ROSTER_SLOTS['TE']}TE/"
    f"{ROSTER_SLOTS['FLEX']}FLEX) are filled from the projection pool, and the baseline is the median of the "
    "three players just past the last startable one — the realistic waiver-wire alternative. Deriving it this "
    "way keeps players and baseline in the same units and lets the baseline follow league settings. "
    f"{LEAGUE_SIZE}-team drafts feature steeper VOR cliffs between rounds due to scarcity and a shallow waiver wire. "
    f"**Round grades** are derived entirely from this model's own VOR rankings — no external ADP is used. "
    f"Overall VOR rank 1–{LEAGUE_SIZE} = Rd 1, {LEAGUE_SIZE+1}–{LEAGUE_SIZE*2} = Rd 2, and so on "
    f"({LEAGUE_SIZE} picks per round for an {LEAGUE_SIZE}-team league). Grades update automatically "
    "whenever projections change, so a breakout candidate rising in VOR will see their round grade improve. "
    "**Expert overlays** applied post-model using live 2026 offseason data: team corrections "
    "(Kyler→MIN, Waddle→DEN, DJ Moore→BUF, Pittman→PIT, Walker→KC, Evans→SF, Etienne→NO, "
    "Henry→BAL, Darnold→SEA, Keenan Allen→LAC, Hopkins→BAL, Dowdle→PIT, Pickens→DAL), "
    "removals (Penix—ACL, Tua—ATL backup, Kamara demoted, Ekeler—Achilles, Nabers—ACL), "
    "force-includes (Kyler Murray—5 games 2025, Willis—MIA starter, Shough—NO starter, Stafford—LAR returning), "
    "age-cliff discounts (Kelce 0.82×, Evans 0.80×, CMC 0.92×), "
    "injury/suspension cuts (Rice 0.70×, Mahomes 0.92×), "
    "curated availability overrides where prior-season games mislead "
    "(Kyler Murray 14g, Jayden Daniels 14g, Daniel Jones 13g—Achilles, Puka Nacua 15g, Rice 10g, Mahomes 14g), "
    "and breakout boosts (Gibbs 1.22×, Skattebo 1.20×, JSN 1.18×, Irving 1.18×, Pickens 1.10×, Pitts 1.10×, Jefferson 1.08×). "
    "**Rookies** (marked **R**) — the 2026 draft class has no NFL history, so the regression engine cannot score them. "
    "Instead they are projected from **draft capital** (overall pick), the single most predictive input for rookie fantasy "
    "production per PlayerProfiler / Dynasty Nerds / PFF research, scaled by depth-chart role (starter / committee / backup / "
    "redshirt). Those base projections then run through the identical team-offense tier, new-HC, and age-curve overlays as "
    "veterans, so landing spot is fully reflected before VOR and round grades are assigned. Draft class lives in "
    "`data/raw/rookies_2026.csv`."
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
