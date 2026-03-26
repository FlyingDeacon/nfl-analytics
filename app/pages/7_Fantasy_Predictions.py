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

# VOR thresholds → fantasy draft round grade (10-team PPR, 10 picks per round)
# Derived directly from actual 2025 model VOR distribution — NOT estimated.
# Each threshold is set at the natural break just below the 10th pick in that round.
#   Rd 1 cutoff (pick 10 → VOR 124.2):  threshold 124
#   Rd 2 cutoff (pick 21 → VOR  93.4):  threshold  93
#   Rd 3 cutoff (pick 31 → VOR  84.4):  threshold  84
#   Rd 4 cutoff (pick 41 → VOR  72.6):  threshold  72
#   Rd 5 cutoff (pick 54 → VOR  65.8):  threshold  65
#   Rd 6 cutoff (pick 67 → VOR  55.9):  threshold  56
ROUND_GRADE_THRESHOLDS = [
    (124, "Rd 1"),   # ~10 players — elite franchise-tier WRs/RBs (picks 1–10)
    ( 93, "Rd 2"),   # ~11 players — high-impact starters (picks 11–21)
    ( 84, "Rd 3"),   # ~10 players — depth starters (picks 22–31)
    ( 72, "Rd 4"),   # ~10 players — value bench depth (picks 32–41)
    ( 65, "Rd 5"),   # ~12 players — fliers and upside plays (picks 42–53)
    ( 56, "Rd 6"),   # ~13 players — deep bench / speculative (picks 54–66)
    (-999,"Rd 7+"),  # Remainder — waiver-wire quality (picks 67+)
]

# Ridge penalty prevents wild extrapolation from small samples.
RIDGE_ALPHA = 8.0
# Per-year recency decay: the 2024→2025 pair is weighted ~35 % higher than 2023→2024.
DECAY = 0.35

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

# Point multipliers based on NFL Expert full 32-team passing/rushing attack audit.
# Values < 1.0 = overvalued / bad team context; > 1.0 = undervalued / great team context.
# Updated March 2026 — all team offensive situations researched.
EXPERT_MULTIPLIERS = {
    # ── Overvalued / bad offensive situation ──────────────────────────────────
    "Travis Kelce":        0.78,  # Age 37 + Mahomes ACL-limited mobility; younger alternatives eating snaps
    "Patrick Mahomes":     0.88,  # ACL limits mobility; offense shifting run-heavy with K. Walker arrival
    "Derrick Henry":       0.90,  # Age regression (32+); BAL run game floor helps but TD pace unsustainable
    "Puka Nacua":          0.87,  # Injury-prone; inconsistent 2024–25 volume
    "Trey McBride":        0.90,  # Regression expected after career-year spike in 2025
    "Rashee Rice":         0.70,  # Suspension ~Wk 7 return; effectively 10-game player (see PROJ_GAMES_OVERRIDES)
    "Mike Evans":          0.82,  # Age 33 (SF) — Purdy boost offsets age concern; keeping slight discount
    "De'Von Achane":       0.80,  # Dolphins full rebuild; Malik Willis unproven + no elite receivers = low-scoring env
    "Devon Achane":        0.80,  # Alt spelling — same player
    "Christian McCaffrey": 0.92,  # Age 30 + back-to-back heavy usage; volume risk but SF offense elite
    "Garrett Wilson":      0.85,  # NYJ/Geno Smith — one of worst QB situations in NFL; WR1 talent wasted
    "Caleb Williams":      0.95,  # DJ Moore traded to BUF; CHI WR corps thin; ceiling reduced despite QB talent
    "Kyle Pitts":          0.87,  # ATL QB instability + franchise tag signals no extension; persistent TE under-usage
    "CeeDee Lamb":         0.88,  # 2025 shoulder injury (75 rec — career low); Pickens target competition
    "Drake London":        0.88,  # Pitts + Darnell Mooney competition; ATL QB uncertainty depresses WR value
    # ── Undervalued / great offensive situation ───────────────────────────────
    "Ja'Marr Chase":       1.15,  # Elite QB-WR duo; best offensive supporting cast in AFC (Burrow + Brown + Higgins)
    "Tee Higgins":         1.12,  # Same CIN elite system; 11 TDs in 2025; Burrow forces ball to both weapons
    "Joe Burrow":          1.10,  # Despite injury history, most complete offense in NFL when healthy
    "Zay Flowers":         1.10,  # Lamar Jackson elite scrambler + DeAndre Hopkins arrival; elite passing env
    "Justin Jefferson":    1.08,  # Kyler Murray + O'Connell system designed for elite WRs; top-5 WR upside
    "Justin Herbert":      1.08,  # Mike McDaniel hired as OC — proven offensive genius; LAC weapons improved
    "Lamar Jackson":       1.12,  # Ravens upgraded weapons (Hopkins + flowers); historical 21+ PPG average
    "Jalen Hurts":         1.08,  # Consistent 20+ PPG; new pass-heavy OC; Brown + Smith + Goedert intact
    "A.J. Brown":          1.05,  # Hurts elite; new pass-heavy OC Mannion; PHI weapons locked in
    "Amon-Ra St. Brown":   1.05,  # First-team All-Pro 2025; new OC Petzing designed offense around him + Williams
    "Drake Maye":          1.15,  # 2024→2025 progression (13.63→20.82 PPG); 3rd-year breakout trajectory
    "DJ Moore":            1.18,  # Josh Allen (elite) + Brady system; Moore thrives with top-5 QB; 1,000+ yds expected
    "Jaxon Smith-Njigba":  1.12,  # Historic $168.6M contract; Darnold improving; run-heavy scheme slightly caps ceiling
    "Jaylen Waddle":       1.10,  # Massive upgrade — Dolphins dysfunction → Bo Nix / DEN passing attack
    "Cam Skattebo":        1.20,  # High-volume starter; model consistently underweights his breakout upside
    "C.J. Stroud":         1.10,  # Bounce-back from shoulder; HOU OL upgraded (Teller); Nico Collins locked in
    "Bucky Irving":        1.10,  # RB1 in TB — Evans departed frees volume; slight PPR concern with Gainwell committee
    "Tucker Kraft":        1.15,  # Ascending TE1 in GB; Jordan Love connection; limited TE competition
    "Kyler Murray":        1.05,  # MIN/Jefferson/O'Connell excellent fit; mobile QB revives offense
    "Kenneth Walker":      1.05,  # Andy Reid system maximizes RB value; KC run-game focus with reduced Mahomes load
    "Michael Pittman":     1.08,  # Reliable target in PIT; upgrade from IND
    "Jahmyr Gibbs":        1.22,  # Solo DET RB; D. Montgomery traded to HOU; massive volume incoming
    "Bijan Robinson":      1.05,  # ATL workhorse — QB instability doesn't hurt elite RBs the same way
    "George Pickens":      1.10,  # DAL move — Dak Prescott elite passer; Pickens is WR1 with elite QB
    "Dak Prescott":        1.06,  # Pickens arrival gives elite deep threat; DAL offense boosted
    "Matthew Stafford":    1.05,  # MVP-caliber 2025 LAR; Adams + Nacua + incoming draft WR depth
    "Rome Odunze":         0.92,  # Forced into CHI WR1 role after Moore traded; more targets but weaker corps
    "Jalen Coker":         1.15,  # Confirmed CAR WR2; ascending role with Bryce Young improvement
}

# 2026 projected games overrides — ONLY for players with confirmed game-count limitations.
# Everyone else defaults to NFL_GAMES (17): healthy starters are assumed to play a full season.
# PPG penalties for injury history are handled separately via EXPERT_MULTIPLIERS.
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
        adj_factor = 1.0 + (proj_games / games_lat.clip(min=1) - 1.0) * 0.45
        pred_pts   = np.clip(raw_pred * adj_factor, 0, None)

        lat = lat.copy()
        lat["predicted_pts"] = pred_pts.round(1)
        lat["proj_games"]    = proj_games.round(1)
        lat["pred_ppg"]      = (pred_pts / proj_games.clip(min=1)).round(2)
        lat["rmse"]          = round(rmse, 1)
        predictions_list.append(lat)

    if not predictions_list:
        return pd.DataFrame(), pd.DataFrame()

    all_preds = pd.concat(predictions_list, ignore_index=True)

    # Historical per-season totals for the trajectory line chart
    hist = (agg.groupby([track_col, name_col, pos_col, "season"], as_index=False)
               [TARGET_COL].sum())
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
                # Find most recent qualifying season
                p_seas = (p_data.groupby("season")[TARGET_COL]
                          .agg(games="count", total_pts="sum")
                          .reset_index())
                qualifying = p_seas[p_seas["games"] >= min_g].sort_values("season", ascending=False)
                if qualifying.empty:
                    continue
                best      = qualifying.iloc[0]
                ppg       = float(best["total_pts"]) / float(best["games"])
                games_2025 = int(p_seas[p_seas["season"] == PREDICTION_YEAR - 1]["games"].sum()
                                 if (PREDICTION_YEAR - 1) in p_seas["season"].values else 0)
                proj_g    = round(min(MAX_PROJ_GAMES,
                                     max(float(min_g),
                                         0.65 * float(best["games"]) + 0.35 * DEFAULT_PROJ_GAMES)), 1)
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

    # 5. Named point multipliers
    for player_fragment, mult in EXPERT_MULTIPLIERS.items():
        mask = out[name_col].str.contains(player_fragment, case=False, na=False)
        if mask.any():
            out.loc[mask, "predicted_pts"] = (out.loc[mask, "predicted_pts"] * mult).round(1)
            out.loc[mask, "pred_ppg"]      = (out.loc[mask, "pred_ppg"] * mult).round(2)

    return out.reset_index(drop=True)


all_preds = apply_expert_adjustments(all_preds_raw, weekly)


def apply_games_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply confirmed 2026 game-count reductions for suspensions / carry-over injuries.

    All other players have already been projected at NFL_GAMES (17) inside build_predictions.
    This function only touches players in PROJ_GAMES_OVERRIDES.
    Predicted points are recalculated as  pred_ppg × new_proj_games  so the per-game
    efficiency (already adjusted by EXPERT_MULTIPLIERS) is preserved exactly.
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
    """Add VOR (Value Over Replacement) and round_grade columns.

    VOR = predicted_pts − replacement_level[position]
    Replacement level is calibrated to a 10-team PPR league (10 picks per round) based on 2025
    championship team analysis. Sorting by VOR rather than raw points accounts
    for positional scarcity — an elite TE ranks higher than an equivalent-points RB.
    """
    def _grade(v: float) -> str:
        for threshold, label in ROUND_GRADE_THRESHOLDS:
            if v >= threshold:
                return label
        return "Rd 7+"

    out = df.copy()
    out["vor"] = out.apply(
        lambda r: round(float(r["predicted_pts"]) - REPLACEMENT_LEVEL.get(r[pos_col], 0), 1),
        axis=1,
    )
    out["round_grade"] = out["vor"].apply(_grade)
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
board_cols += ["predicted_pts", "vor", "round_grade", "pred_ppg", "proj_games", "last_season_pts", "change", "change_pct", "games"]

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
    "predicted_pts": "2026 Proj",
    "vor":          "VOR",
    "round_grade":  "Round",
    "pred_ppg": "Proj PPG",
    "proj_games": "Proj GP",
    "last_season_pts": "2025 Actual",
    "change": "Δ Pts",
    "change_pct": "Δ %",
    "games": "2025 GP",
    "passing_yards": "Pass Yds", "passing_tds": "Pass TD", "interceptions": "INT",
    "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD", "carries": "Carries",
    "receptions": "Rec",   "targets": "Tgt",
    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
}
if team_col: rename_map[team_col] = "Team"
if pos_col:  rename_map[pos_col]  = "Pos"

disp = preds[board_cols].copy()
for c in disp.select_dtypes("float").columns:
    if c != "change_pct":
        disp[c] = disp[c].round(1)

# Add logo URLs for team column if it exists
teams_df = load_teams()
column_config_dict = {
    "2026 Proj":  st.column_config.NumberColumn(format="%.1f"),
    "VOR":        st.column_config.NumberColumn(format="%.1f",
                      help="Value Over Replacement — positional scarcity-adjusted score. "
                           "Accounts for how scarce elite players are at each position."),
    "Round":      st.column_config.TextColumn(help="Suggested fantasy draft round (10-team PPR)"),
    "Proj PPG":   st.column_config.NumberColumn(format="%.2f"),
    "Δ Pts":      st.column_config.NumberColumn(format="%+.1f"),
    "Δ %":        st.column_config.NumberColumn(format="%+.1f%%"),
}

if team_col:
    disp["_logo_url"] = disp[team_col].apply(lambda t: get_logo(t, teams_df) if pd.notna(t) else "")
    column_config_dict["_logo_url"] = st.column_config.ImageColumn(
        label="",
        width="small",
    )

disp_renamed = disp.rename(columns=rename_map)
if "_logo_url" in disp.columns:
    # Reorder to put logo before Team
    cols = list(disp_renamed.columns)
    if "_logo_url" in cols:
        cols.remove("_logo_url")
        cols.insert(cols.index("Team"), "_logo_url")
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
    f"TE={REPLACEMENT_LEVEL['TE']}) calibrated from 2025 championship team data. 10-team drafts feature "
    "steeper VOR cliffs between rounds due to scarcity and a shallow waiver wire. "
    "**Round grades** derived from actual 2025 model VOR distribution (10-team PPR, 10 picks/round): "
    "Rd 1 (VOR≥124)=~10 elite franchise WRs/RBs (picks 1–10); "
    "Rd 2 (93–123)=~11 high-impact starters (picks 11–21); "
    "Rd 3 (84–92)=~10 depth starters (picks 22–31); "
    "Rd 4 (72–83)=~10 bench value picks (picks 32–41); "
    "Rd 5 (65–71)=~12 fliers and upside plays (picks 42–53); "
    "Rd 6 (56–64)=~13 deep bench/speculative (picks 54–66); "
    "Rd 7+ (<56)=waiver-wire quality (picks 67+). "
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

        fig.add_trace(go.Scatter(
            x=szns, y=vals, mode="lines+markers", name=player,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(color="#fff", width=1)),
            hovertemplate=f"<b>{player}</b><br>%{{x}}: %{{y:,.1f}} pts<extra></extra>",
            legendgroup=player,
        ))
        fig.add_trace(go.Scatter(
            x=[szns[-1], PREDICTION_YEAR], y=[vals[-1], pred_val],
            mode="lines+markers", showlegend=False, legendgroup=player,
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=10, color=color, symbol="star",
                        line=dict(color="#fff", width=1.5)),
            hovertemplate=f"<b>{player}</b><br>2026 Proj: {pred_val:,.1f} pts<extra></extra>",
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
