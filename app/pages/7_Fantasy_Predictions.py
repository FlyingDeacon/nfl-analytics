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

# Players removed from 2026 board (not projected starters / retired / injury)
EXPERT_REMOVE = {
    "Kirk Cousins",      # Not a projected 2026 starter
    "Matthew Stafford",  # Aging (38), likely retired or backup
}

# Team corrections: player name fragment → corrected 2026 team abbreviation
EXPERT_TEAM_CORRECTIONS = {
    "Travis Etienne": "NO",   # Signed with New Orleans Saints (left JAX)
}

# Point multipliers based on NFL Expert contextual analysis.
# Values < 1.0 = overvalued by the model; > 1.0 = undervalued.
EXPERT_MULTIPLIERS = {
    # ── Overvalued ────────────────────────────────────────────────────────────
    "Travis Kelce":       0.82,   # Age cliff (36 in 2026); declining target share
    "Derrick Henry":      0.85,   # RB age regression (32+); carry accumulation
    "Patrick Mahomes":    0.95,   # Conservative ACL recovery / workload discount
    "Puka Nacua":         0.87,   # Injury-prone; disappointing 2024 volume
    "Trey McBride":       0.90,   # Regression expected after career-year spike
    "Drake Maye":         0.88,   # Year-2 uncertainty; thin supporting cast
    "Josh Allen":         0.95,   # Mild regression signal; still top-tier
    # ── Undervalued ───────────────────────────────────────────────────────────
    "Garrett Wilson":     1.12,   # Elite route runner; improved QB situation
    "Jaxon Smith-Njigba": 1.15,   # WR1 role fully cemented; breakout expected
    "Bucky Irving":       1.18,   # Projected RB1 in Tampa Bay
    "Cam Skattebo":       1.20,   # High-volume Giants starter; data underweights him
    "C.J. Stroud":        1.10,   # Bounce-back from shoulder injury
    "Tucker Kraft":       1.15,   # Ascending TE1 in Green Bay
    "Rashee Rice":        0.70,   # Suspension — available ~Week 7+ only (~10 games)
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
top_n = st.sidebar.slider("Big Board Size", 10, 100, 50, key=f"pred_top_{_v}")

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
        # Blend last two seasons' games to smooth out injury anomalies.
        # A player who played 8/17 games likely recovers toward ~13 next year.
        games_lat = lat["games"].values.astype(float)
        if prev_szn is not None:
            prev_map = (pos_df[pos_df["season"] == prev_szn]
                        .drop_duplicates(track_col)
                        .set_index(track_col)["games"])
            games_prev = np.array([
                float(prev_map.loc[pid]) if pid in prev_map.index else DEFAULT_PROJ_GAMES
                for pid in lat[track_col].values
            ])
        else:
            games_prev = np.full(len(lat), DEFAULT_PROJ_GAMES)

        proj_games = np.clip(
            0.65 * games_lat + 0.35 * games_prev,
            a_min=float(min_g),
            a_max=MAX_PROJ_GAMES,
        )

        # Damped games-ratio adjustment (don't fully extrapolate injured players)
        # A player who played 8 games → projected 13: adjustment ≈ +31 % of gap
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


def apply_expert_adjustments(df: pd.DataFrame) -> pd.DataFrame:
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

    # 3. Team corrections (trades / FA signings not captured in historical data)
    if team_col:
        for player_fragment, new_team in EXPERT_TEAM_CORRECTIONS.items():
            mask = out[name_col].str.contains(player_fragment, case=False, na=False)
            out.loc[mask, team_col] = new_team

    # 4. Named point multipliers
    for player_fragment, mult in EXPERT_MULTIPLIERS.items():
        mask = out[name_col].str.contains(player_fragment, case=False, na=False)
        if mask.any():
            out.loc[mask, "predicted_pts"] = (out.loc[mask, "predicted_pts"] * mult).round(1)
            out.loc[mask, "pred_ppg"]      = (out.loc[mask, "pred_ppg"] * mult).round(2)

    return out.reset_index(drop=True)


all_preds = apply_expert_adjustments(all_preds_raw)

if all_preds.empty:
    st.error("Not enough historical data to build predictions.")
    st.stop()

# ── Filter by position ───────────────────────────────────────────────────────
preds = (all_preds[all_preds[pos_col] == sel_pos].copy()
         if sel_pos != "All" else all_preds.copy())
preds = preds.sort_values("predicted_pts", ascending=False).head(top_n).reset_index(drop=True)
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

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 2026 BIG BOARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### 📋 2026 Fantasy Big Board {pos_label_str}")

board_cols = ["Rank", name_col]
if team_col: board_cols.append(team_col)
if pos_col:  board_cols.append(pos_col)
board_cols += ["predicted_pts", "pred_ppg", "proj_games", "last_season_pts", "change", "change_pct", "games"]

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

st.dataframe(
    disp.rename(columns=rename_map),
    hide_index=True,
    use_container_width=True,
    column_config={
        "2026 Proj":  st.column_config.NumberColumn(format="%.1f"),
        "Proj PPG":   st.column_config.NumberColumn(format="%.2f"),
        "Δ Pts":      st.column_config.NumberColumn(format="%+.1f"),
        "Δ %":        st.column_config.NumberColumn(format="%+.1f%%"),
    },
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RISERS & FALLERS — per-position % change so QBs don't dominate raw-point deltas
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Risers &nbsp;&nbsp; 📉 Fallers")
st.caption("Ranked by % change within each position — QBs and skill positions compared fairly on relative improvement.")

positions_to_show = [sel_pos] if sel_pos != "All" else list(POSITION_FEATURES.keys())

def _rf_table_html(df: pd.DataFrame, is_riser: bool) -> str:
    """Render a risers/fallers DataFrame as a crisp HTML table (no canvas → never blurry)."""
    arrow = "▲" if is_riser else "▼"
    hdr_color = "#10b981" if is_riser else "#ef4444"
    rows_html = ""
    for _, row in df.iterrows():
        delta_pts = row.get("Δ Pts", 0)
        delta_pct = row.get("Δ %",  0)
        color = "#10b981" if float(delta_pts) >= 0 else "#ef4444"
        sign  = "+" if float(delta_pts) >= 0 else ""
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;'>{row.get('Player','')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;color:#8b8fa8;'>{row.get('Team','')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;text-align:right;font-weight:600;'>{row.get('2026 Proj','')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;text-align:right;color:#8b8fa8;'>{row.get('2025 Pts','')}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;text-align:right;color:{color};font-weight:700;'>"
            f"{sign}{delta_pts:+.1f}</td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #2a2d3e;text-align:right;color:{color};font-weight:700;'>"
            f"{sign}{delta_pct:+.1f}%</td>"
            f"</tr>"
        )
    th = (
        "<thead><tr style='background:#1a1d2e;'>"
        + "".join(
            f"<th style='padding:8px 10px;text-align:{'right' if i>1 else 'left'};"
            f"font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;"
            f"color:{hdr_color if col in ('Δ Pts','Δ %') else '#8b8fa8'};white-space:nowrap;'>{col}</th>"
            for i, col in enumerate(["Player", "Team", "2026 Proj", "2025 Pts", "Δ Pts", "Δ %"])
        )
        + "</tr></thead>"
    )
    return (
        f"<table style='width:100%;border-collapse:collapse;font-size:0.88rem;"
        f"background:#12152a;border-radius:8px;overflow:hidden;'>"
        f"{th}<tbody>{rows_html}</tbody></table>"
    )


for pos in positions_to_show:
    pos_preds = all_preds[all_preds[pos_col] == pos].copy()
    pos_preds["last_season_pts"] = pos_preds[TARGET_COL].round(1)
    pos_preds["change"]          = (pos_preds["predicted_pts"] - pos_preds["last_season_pts"]).round(1)
    pos_preds["change_pct"]      = ((pos_preds["change"] /
                                     pos_preds["last_season_pts"].replace(0, np.nan)) * 100).round(1)
    pos_preds = pos_preds.dropna(subset=["change_pct"])

    if pos_preds.empty:
        continue

    rise_cols = [name_col, team_col, "predicted_pts", "last_season_pts", "change", "change_pct"]
    rise_cols = [c for c in rise_cols if c in pos_preds.columns]
    rise_rename = {name_col: "Player", team_col: "Team",
                   "predicted_pts": "2026 Proj", "last_season_pts": "2025 Pts",
                   "change": "Δ Pts", "change_pct": "Δ %"}

    risers  = pos_preds.nlargest(5,  "change_pct")[rise_cols].rename(columns=rise_rename)
    fallers = pos_preds.nsmallest(5, "change_pct")[rise_cols].rename(columns=rise_rename)

    st.markdown(f"**{POSITION_LABELS[pos]}**")
    # Render as HTML tables — avoids canvas/AG Grid blur inside narrow columns
    st.markdown(
        f'<div style="display:flex;gap:16px;margin-bottom:20px;">'
        f'<div style="flex:1;">'
        f'<div style="font-size:0.8rem;color:#10b981;font-weight:700;margin-bottom:6px;">📈 Risers</div>'
        f'{_rf_table_html(risers, is_riser=True)}'
        f'</div>'
        f'<div style="flex:1;">'
        f'<div style="font-size:0.8rem;color:#ef4444;font-weight:700;margin-bottom:6px;">📉 Fallers</div>'
        f'{_rf_table_html(fallers, is_riser=False)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "**Methodology** — Position-specific ridge regression trained on 2016–2025 consecutive-season pairs "
    "with exponential recency weighting (recent seasons count more). Features are per-game rates, not "
    "season totals, so a player who missed games due to injury is not penalised for low counting stats. "
    "Projected 2026 games blends the last two seasons (65 / 35 weighting) with a conservative ceiling of "
    f"{MAX_PROJ_GAMES} games. QB qualifier: {MIN_GAMES_BY_POS['QB']}+ games started. "
    "Skill positions: 6+ games. "
    "**Expert overlays** applied post-model: age-cliff discounts (Kelce, Henry), injury/suspension adjustments "
    "(Rice, Mahomes), team corrections (Etienne → NO), and named boosts for undervalued breakout candidates "
    "(Skattebo, Irving, JSN, Tucker Kraft)."
)
