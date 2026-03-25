import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Fantasy Predictions · NFL", page_icon="🔮", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="7_Fantasy_Predictions")

# ── Back button ──────────────────────────────────────────────────────────────
if st.button("← Back to Fantasy Football", key="back_btn", help="Return to Fantasy Football"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("<style>div[data-testid='stButton']:has(button:contains('Back')) { max-width: 180px; }</style>", unsafe_allow_html=True)

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔮</div>
    <div>
        <div class="title">2026 Fantasy Predictions</div>
        <div class="subtitle">Machine-learning projections powered by multiple linear regression</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Multiple Linear Regression (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Ordinary least-squares via the normal equation.
    Returns (coefficients, intercept, R² score)."""
    n = X.shape[0]
    X_b = np.column_stack([np.ones(n), X])           # add bias column
    theta, residuals, rank, sv = np.linalg.lstsq(X_b, y, rcond=None)
    intercept = theta[0]
    coefs = theta[1:]
    y_pred = X_b @ theta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coefs, intercept, r2


def _ols_predict(X: np.ndarray, coefs: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coefs + intercept


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — position-specific feature sets (tuned by data analysis)
# ══════════════════════════════════════════════════════════════════════════════

POSITION_FEATURES = {
    "QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards",
           "rushing_tds", "completions", "attempts"],
    "RB": ["rushing_yards", "rushing_tds", "carries", "receptions",
           "receiving_yards", "receiving_tds", "targets"],
    "WR": ["receiving_yards", "receiving_tds", "targets", "receptions",
           "rushing_yards", "receiving_air_yards", "receiving_yards_after_catch"],
    "TE": ["receiving_yards", "receiving_tds", "targets", "receptions",
           "receiving_air_yards", "receiving_yards_after_catch"],
}

POSITION_LABELS = {"QB": "Quarterbacks", "RB": "Running Backs",
                   "WR": "Wide Receivers", "TE": "Tight Ends"}

MIN_GAMES = 8          # Minimum games in a season to qualify
PREDICTION_YEAR = 2026  # Target prediction season
TARGET_COL = "fantasy_points_ppr"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

weekly = load_weekly()
teams  = load_teams()

if weekly.empty:
    st.warning("No weekly player data found. Run load_nfl_data.py first.")
    st.stop()

# Column detection
name_col = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
id_col   = next((c for c in ["player_id", "gsis_id"] if c in weekly.columns), None)
team_col = next((c for c in ["recent_team", "posteam", "team"] if c in weekly.columns), None)
pos_col  = next((c for c in ["position", "pos"] if c in weekly.columns), None)

if not name_col or not pos_col:
    st.error("Required columns (player name, position) not found in data.")
    st.stop()

# Use player_id for tracking across seasons; fall back to name
track_col = id_col if id_col else name_col

# ── Sidebar filters ──────────────────────────────────────────────────────────
if "pred_v" not in st.session_state:
    st.session_state["pred_v"] = 0
_v = st.session_state["pred_v"]

sel_pos = st.sidebar.selectbox("Position", ["All"] + list(POSITION_FEATURES.keys()),
                               key=f"pred_pos_{_v}")
top_n = st.sidebar.slider("Big Board Size", 10, 100, 50, key=f"pred_top_{_v}")
show_model_details = st.sidebar.checkbox("Show Model Details", value=False, key=f"pred_details_{_v}")

if st.sidebar.button("Reset Filters", key="pred_reset", use_container_width=True):
    st.session_state["pred_v"] = _v + 1
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SEASON AGGREGATES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building prediction models …")
def build_predictions(weekly_df: pd.DataFrame):
    """Train position-specific regression models and predict 2026."""

    positions_to_model = list(POSITION_FEATURES.keys())
    all_seasons = sorted(weekly_df["season"].dropna().unique().astype(int))

    # -- Step 1: aggregate season totals per player --
    group_keys = [track_col, name_col, pos_col, "season"]
    if team_col:
        group_keys.append(team_col)

    # Sum all possible feature columns + target
    all_feat_cols = list({c for feats in POSITION_FEATURES.values() for c in feats})
    stat_cols = [TARGET_COL] + [c for c in all_feat_cols if c in weekly_df.columns]

    # Only regular season
    reg = weekly_df.copy()
    if "season_type" in reg.columns:
        reg = reg[reg["season_type"] == "REG"]

    reg = reg[reg[pos_col].isin(positions_to_model)]

    # Aggregate + games played
    agg = reg.groupby(group_keys, as_index=False)[stat_cols].sum()
    gp  = reg.groupby(group_keys, as_index=False)[TARGET_COL].count()
    gp.rename(columns={TARGET_COL: "games"}, inplace=True)
    agg = agg.merge(gp, on=group_keys, how="left")
    agg = agg[agg["games"] >= MIN_GAMES]

    # Per-game rates (used as additional features)
    for c in stat_cols:
        if c in agg.columns:
            agg[f"{c}_pg"] = (agg[c] / agg["games"]).round(3)

    # -- Step 2: train per-position models --
    model_results = {}
    predictions_list = []

    for pos in positions_to_model:
        feat_names = [c for c in POSITION_FEATURES[pos] if c in agg.columns]
        # Use both totals and per-game rates as features (doubles feature space)
        feat_total = feat_names
        feat_pg = [f"{c}_pg" for c in feat_names if f"{c}_pg" in agg.columns]
        all_feats = feat_total + feat_pg + ["games"]

        pos_df = agg[agg[pos_col] == pos].copy()

        # Build training pairs: season N features → season N+1 target
        train_rows = []
        for szn in all_seasons[:-1]:
            next_szn = szn + 1
            if next_szn not in all_seasons:
                continue
            cur = pos_df[pos_df["season"] == szn].set_index(track_col)
            nxt = pos_df[pos_df["season"] == next_szn].set_index(track_col)
            common = cur.index.intersection(nxt.index)
            for pid in common:
                row_cur = cur.loc[pid] if isinstance(cur.loc[pid], pd.Series) else cur.loc[pid].iloc[-1]
                row_nxt = nxt.loc[pid] if isinstance(nxt.loc[pid], pd.Series) else nxt.loc[pid].iloc[-1]
                x_vals = [float(row_cur.get(f, 0)) for f in all_feats]
                y_val  = float(row_nxt[TARGET_COL])
                train_rows.append(x_vals + [y_val])

        if len(train_rows) < 20:
            model_results[pos] = {"r2": 0, "n_train": len(train_rows),
                                   "features": all_feats, "coefs": None, "intercept": 0}
            continue

        train_arr = np.array(train_rows, dtype=np.float64)
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]

        # Standardize features for numerical stability
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_scaled = (X_train - mu) / sigma

        coefs, intercept, r2 = _ols_fit(X_scaled, y_train)

        model_results[pos] = {
            "r2": r2, "n_train": len(train_rows),
            "features": all_feats, "coefs": coefs, "intercept": intercept,
            "mu": mu, "sigma": sigma,
        }

        # -- Step 3: predict 2026 using latest available season data --
        latest_season = all_seasons[-1]
        latest_pos = pos_df[pos_df["season"] == latest_season].copy()

        if latest_pos.empty:
            continue

        X_latest = latest_pos[all_feats].fillna(0).values.astype(np.float64)
        X_latest_scaled = (X_latest - mu) / sigma
        preds = _ols_predict(X_latest_scaled, coefs, intercept)
        preds = np.clip(preds, 0, None)  # floor at 0

        latest_pos["predicted_pts"] = preds.round(1)
        latest_pos["pred_ppg"] = (latest_pos["predicted_pts"] / 17).round(1)  # 17-game season
        predictions_list.append(latest_pos)

    if predictions_list:
        all_preds = pd.concat(predictions_list, ignore_index=True)
    else:
        all_preds = pd.DataFrame()

    # -- Step 4: build historical yearly totals for line chart --
    hist = agg.groupby([track_col, name_col, pos_col, "season"], as_index=False)[TARGET_COL].sum()
    if team_col and team_col in agg.columns:
        team_lookup = agg.groupby([track_col, "season"], as_index=False)[team_col].first()
        hist = hist.merge(team_lookup, on=[track_col, "season"], how="left")

    return all_preds, model_results, hist


all_preds, model_results, hist_totals = build_predictions(weekly)

if all_preds.empty:
    st.error("Not enough historical data to build predictions.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# FILTER BY SELECTED POSITION
# ══════════════════════════════════════════════════════════════════════════════

if sel_pos != "All":
    preds = all_preds[all_preds[pos_col] == sel_pos].copy()
else:
    preds = all_preds.copy()

preds = preds.sort_values("predicted_pts", ascending=False).head(top_n).reset_index(drop=True)
preds.insert(0, "Rank", range(1, len(preds) + 1))

if preds.empty:
    st.info("No predictions available for the selected position.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ACCURACY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Model Performance")

mcols = st.columns(len(model_results))
for i, (pos, info) in enumerate(model_results.items()):
    with mcols[i]:
        r2_pct = info["r2"] * 100
        color = "#10b981" if r2_pct >= 45 else "#f59e0b" if r2_pct >= 30 else "#ef4444"
        st.markdown(f"""
        <div class="stat-card" style="text-align:center; padding:18px 10px;">
            <div class="label">{POSITION_LABELS[pos]}</div>
            <div class="value" style="font-size:1.8rem; color:{color};">{r2_pct:.1f}%</div>
            <div class="sub">R² accuracy · {info['n_train']:,} training pairs</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DETAILS (expandable)
# ══════════════════════════════════════════════════════════════════════════════

if show_model_details:
    st.markdown("### Model Details")
    st.caption("Multiple linear regression trained on consecutive season pairs. "
               "Features include season totals, per-game rates, and games played. "
               "Models are position-specific to capture role differences.")
    for pos, info in model_results.items():
        with st.expander(f"{POSITION_LABELS[pos]} — Features & Coefficients"):
            if info["coefs"] is not None:
                feat_df = pd.DataFrame({
                    "Feature": info["features"],
                    "Coefficient (scaled)": info["coefs"].round(4),
                })
                feat_df["Abs Impact"] = feat_df["Coefficient (scaled)"].abs()
                feat_df = feat_df.sort_values("Abs Impact", ascending=False)
                st.dataframe(feat_df.drop(columns=["Abs Impact"]),
                             hide_index=True, use_container_width=True)
                st.caption(f"R² = {info['r2']:.4f} · {info['n_train']} training samples · "
                           f"Features standardized (zero-mean, unit-variance)")
            else:
                st.warning(f"Not enough data to train {pos} model.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TOP 3 PREDICTED PLAYERS — HERO CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### 🏆 2026 Projected Top 3 {'(' + sel_pos + ')' if sel_pos != 'All' else '(Overall)'}")

if len(preds) >= 3:
    c1, c2, c3 = st.columns(3)
    for i, (col, medal) in enumerate(zip([c1, c2, c3], ["🥇 #1", "🥈 #2", "🥉 #3"])):
        row = preds.iloc[i]
        player = row[name_col]
        team_abbr = row[team_col] if team_col else "—"
        pos  = row[pos_col] if pos_col else ""
        pred_pts = row["predicted_pts"]
        pred_ppg = row["pred_ppg"]
        last_pts = row[TARGET_COL]
        delta = pred_pts - last_pts
        delta_sign = "+" if delta >= 0 else ""
        logo_html = ""
        url = get_logo(team_abbr, teams)
        if url:
            logo_html = f'<img src="{url}" width="40" style="margin:6px 0;">'
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">{medal}</div>
                {logo_html}
                <div class="value" style="font-size:1.15rem;">{player}</div>
                <div class="sub">{team_abbr} · {pos}</div>
                <div style="font-size:1.5rem; font-weight:800; color:var(--gold); margin:8px 0;">
                    {pred_pts:,.1f} <span style="font-size:0.8rem; font-weight:400;">proj pts</span>
                </div>
                <div class="sub">{pred_ppg} PPG · <span style="color:{'#10b981' if delta >= 0 else '#ef4444'};">{delta_sign}{delta:,.1f} vs last season</span></div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LINE CHART — Historical + Predicted 2026
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Historical Trajectory + 2026 Projection")

# Let user pick players for the line chart
top10_names = preds.head(10)[name_col].tolist()
chart_players = st.multiselect(
    "Select players to chart",
    options=preds[name_col].tolist(),
    default=top10_names[:5],
    key=f"pred_chart_players_{_v}",
)

if chart_players:
    fig = go.Figure()

    for player in chart_players:
        # Get track_col value for this player
        player_pred = preds[preds[name_col] == player]
        if player_pred.empty:
            continue
        pid = player_pred.iloc[0][track_col]
        team_abbr = player_pred.iloc[0][team_col] if team_col else ""
        color = TEAM_COLORS.get(team_abbr, "#4f46e5")
        pred_val = player_pred.iloc[0]["predicted_pts"]

        # Historical data
        player_hist = hist_totals[hist_totals[track_col] == pid].sort_values("season")

        if player_hist.empty:
            continue

        seasons = player_hist["season"].tolist()
        values  = player_hist[TARGET_COL].tolist()

        # Solid line for historical
        fig.add_trace(go.Scatter(
            x=seasons, y=values,
            mode="lines+markers",
            name=player,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color, line=dict(color="#fff", width=1)),
            hovertemplate=f"<b>{player}</b><br>Season %{{x}}<br>PPR: %{{y:,.1f}}<extra></extra>",
            legendgroup=player,
        ))

        # Dashed line from last historical point to 2026 prediction
        last_season = seasons[-1]
        last_val    = values[-1]
        fig.add_trace(go.Scatter(
            x=[last_season, PREDICTION_YEAR],
            y=[last_val, pred_val],
            mode="lines+markers",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=10, color=color, symbol="star",
                        line=dict(color="#fff", width=1.5)),
            hovertemplate=(f"<b>{player}</b><br>2026 Projection<br>"
                           f"PPR: {pred_val:,.1f}<extra></extra>"),
            showlegend=False,
            legendgroup=player,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Fantasy PPR Points — Historical + 2026 Projection",
        xaxis_title="Season",
        yaxis_title="Total Fantasy Points (PPR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis=dict(dtick=1),
    )

    # Add a subtle annotation for the prediction zone
    fig.add_vrect(
        x0=max(hist_totals["season"].max(), 2025) + 0.5,
        x1=PREDICTION_YEAR + 0.5,
        fillcolor="#4f46e5", opacity=0.05,
        layer="below", line_width=0,
        annotation_text="Projected", annotation_position="top left",
        annotation_font_color="#888",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one player above to see the trajectory chart.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# 2026 BIG BOARD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### 📋 2026 Fantasy Big Board {'(' + sel_pos + ')' if sel_pos != 'All' else ''}")

# Compute year-over-year change
preds["last_season_pts"] = preds[TARGET_COL].round(1)
preds["change"] = (preds["predicted_pts"] - preds["last_season_pts"]).round(1)
preds["change_pct"] = ((preds["change"] / preds["last_season_pts"].replace(0, np.nan)) * 100).round(1)

# Build display table
board_cols = ["Rank", name_col]
if team_col: board_cols.append(team_col)
if pos_col:  board_cols.append(pos_col)
board_cols += ["predicted_pts", "pred_ppg", "last_season_pts", "change", "change_pct", "games"]

# Add key stats based on position
if sel_pos == "QB" or sel_pos == "All":
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
    "predicted_pts": "2026 Proj Pts",
    "pred_ppg": "Proj PPG",
    "last_season_pts": "Last Season Pts",
    "change": "Δ Pts",
    "change_pct": "Δ %",
    "games": "GP (Last)",
    "passing_yards": "Pass Yds", "passing_tds": "Pass TD",
    "interceptions": "INT", "rushing_yards": "Rush Yds",
    "rushing_tds": "Rush TD", "carries": "Carries",
    "receptions": "Rec", "targets": "Tgt",
    "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
}
if team_col: rename_map[team_col] = "Team"
if pos_col:  rename_map[pos_col] = "Pos"

disp = preds[board_cols].copy()
for c in disp.select_dtypes("float").columns:
    if c in ("change_pct",):
        continue
    disp[c] = disp[c].round(1)

st.dataframe(
    disp.rename(columns=rename_map),
    hide_index=True,
    use_container_width=True,
    column_config={
        "2026 Proj Pts": st.column_config.NumberColumn(format="%.1f"),
        "Proj PPG": st.column_config.NumberColumn(format="%.1f"),
        "Δ Pts": st.column_config.NumberColumn(format="%+.1f"),
        "Δ %": st.column_config.NumberColumn(format="%+.1f%%"),
    },
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# POSITION BREAKDOWN BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Projected Points by Position")

pos_breakdown = preds.head(top_n)
bar_colors = [TEAM_COLORS.get(row.get(team_col, ""), "#4f46e5") for _, row in pos_breakdown.iterrows()]

fig_bar = px.bar(
    pos_breakdown,
    x=name_col,
    y="predicted_pts",
    color=pos_col if pos_col else None,
    hover_data=["pred_ppg", "last_season_pts", "change"],
    color_discrete_map={"QB": "#E31837", "RB": "#003594", "WR": "#10b981", "TE": "#f59e0b"},
)
fig_bar.update_layout(
    **PLOTLY_LAYOUT,
    title=f"2026 Projected Fantasy Points{' — ' + sel_pos if sel_pos != 'All' else ''}",
    xaxis_title="Player",
    yaxis_title="Projected PPR Points",
    showlegend=True if sel_pos == "All" else False,
    xaxis_tickangle=-35,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Add team logos above bars
y_max = pos_breakdown["predicted_pts"].max()
logo_h = y_max * 0.10
if team_col:
    for idx, (_, row) in enumerate(pos_breakdown.iterrows()):
        url = get_logo(row[team_col], teams)
        if url:
            fig_bar.add_layout_image(dict(
                source=url, xref="x", yref="y",
                x=idx, y=row["predicted_pts"] + y_max * 0.01,
                sizex=0.8, sizey=logo_h,
                xanchor="center", yanchor="bottom", layer="above",
            ))
    fig_bar.update_layout(yaxis_range=[0, y_max * 1.18])

st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RISERS & FALLERS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Biggest Risers & Fallers")

r_col, f_col = st.columns(2)

with r_col:
    st.markdown("#### 📈 Risers")
    risers = preds.nlargest(10, "change")[["Rank", name_col, pos_col, "predicted_pts",
                                            "last_season_pts", "change"]].copy()
    risers = risers.rename(columns={name_col: "Player", pos_col: "Pos",
                                     "predicted_pts": "2026 Proj",
                                     "last_season_pts": "Last Szn",
                                     "change": "Δ Pts"})
    st.dataframe(risers, hide_index=True, use_container_width=True,
                 column_config={"Δ Pts": st.column_config.NumberColumn(format="%+.1f")})

with f_col:
    st.markdown("#### 📉 Fallers")
    fallers = preds.nsmallest(10, "change")[["Rank", name_col, pos_col, "predicted_pts",
                                              "last_season_pts", "change"]].copy()
    fallers = fallers.rename(columns={name_col: "Player", pos_col: "Pos",
                                       "predicted_pts": "2026 Proj",
                                       "last_season_pts": "Last Szn",
                                       "change": "Δ Pts"})
    st.dataframe(fallers, hide_index=True, use_container_width=True,
                 column_config={"Δ Pts": st.column_config.NumberColumn(format="%+.1f")})

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Projections generated using position-specific multiple linear regression models trained on "
           "consecutive season pairs (2016–2025). Features include season totals, per-game rates, and "
           "games played. Minimum 8 games required to qualify. Predictions assume a 17-game season.")
