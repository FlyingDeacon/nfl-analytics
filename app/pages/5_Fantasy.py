import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_weekly, load_teams, get_logo, load_preseason_rankings, _normalize_name
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Fantasy Football · NFL", page_icon="🏆", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="5_Fantasy")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🏆</div>
    <div>
        <div class="title">Fantasy Football</div>
        <div class="subtitle">Season totals, per-game averages, and weekly consistency</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

weekly = load_weekly()
teams  = load_teams()

if weekly.empty:
    st.warning("No weekly player data found. Run load_nfl_data.py first.")
    st.stop()

# ── Column detection ─────────────────────────────────────────────────────────
name_col = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
team_col = next((c for c in ["recent_team", "posteam", "team"] if c in weekly.columns), None)
pos_col  = next((c for c in ["position", "pos"] if c in weekly.columns), None)

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K"]

# ── Sidebar filters ──────────────────────────────────────────────────────────
if "fan_v" not in st.session_state:
    st.session_state["fan_v"] = 0
_v = st.session_state["fan_v"]

seasons = sorted(weekly["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key=f"fan_season_{_v}")

scoring = st.sidebar.radio("Scoring Format", ["PPR", "Standard", "Half PPR"], key=f"fan_scoring_{_v}")
pts_col = "fantasy_points_ppr" if scoring == "PPR" else "fantasy_points"

sel_pos = st.sidebar.selectbox("Position", ["All"] + FANTASY_POSITIONS, key=f"fan_pos_{_v}")

top_n = st.sidebar.slider("Show top N players", 5, 50, 25, key=f"fan_top_n_{_v}")

# Player autocomplete — scoped to selected season & position
_player_pool = weekly[weekly["season"] == sel_season].copy()
if pos_col and sel_pos != "All":
    _player_pool = _player_pool[_player_pool[pos_col] == sel_pos]
if name_col:
    _season_players = sorted(_player_pool[name_col].dropna().unique().tolist())
    sel_player = st.sidebar.selectbox(
        "Search Player", ["All Players"] + _season_players, key=f"fan_player_{_v}"
    )
else:
    sel_player = "All Players"

if st.sidebar.button("Reset Filters", key="fan_reset", use_container_width=True):
    st.session_state["fan_v"] = _v + 1
    st.rerun()

# ── Filter raw weekly data ───────────────────────────────────────────────────
df = weekly[weekly["season"] == sel_season].copy()

# Keep only fantasy-relevant positions
if pos_col:
    df = df[df[pos_col].isin(FANTASY_POSITIONS)]

if sel_pos != "All" and pos_col:
    df = df[df[pos_col] == sel_pos]

if sel_player != "All Players" and name_col:
    df = df[df[name_col] == sel_player]

if df.empty:
    st.info("No data for the selected filters.")
    st.stop()

# Half PPR: average of standard and PPR
if scoring == "Half PPR" and "fantasy_points" in df.columns and "fantasy_points_ppr" in df.columns:
    df["half_ppr"] = ((df["fantasy_points"] + df["fantasy_points_ppr"]) / 2).round(2)
    pts_col = "half_ppr"

# ── Aggregate season totals ──────────────────────────────────────────────────
group_cols = [name_col]
if team_col: group_cols.append(team_col)
if pos_col:  group_cols.append(pos_col)

stat_cols = [pts_col]
extra_cols = ["passing_yards", "passing_tds", "interceptions",
              "rushing_yards", "rushing_tds", "carries",
              "receptions", "targets", "receiving_yards", "receiving_tds"]
stat_cols += [c for c in extra_cols if c in df.columns]

# Season totals
agg = df.groupby(group_cols, as_index=False)[stat_cols].sum()

# Games played & per-game averages
games = df.groupby(group_cols, as_index=False)[pts_col].count()
games.rename(columns={pts_col: "games"}, inplace=True)
agg = agg.merge(games, on=group_cols, how="left")
agg["ppg"] = (agg[pts_col] / agg["games"]).round(1)

# Total TDs
td_cols = [c for c in ["passing_tds", "rushing_tds", "receiving_tds"] if c in agg.columns]
agg["total_tds"] = agg[td_cols].sum(axis=1).astype(int)

# Total fumbles lost
fum_cols = [c for c in ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]
            if c in df.columns]
if fum_cols:
    fum = df.groupby(group_cols, as_index=False)[fum_cols].sum()
    fum["fumbles_lost"] = fum[fum_cols].sum(axis=1).astype(int)
    agg = agg.merge(fum[group_cols + ["fumbles_lost"]], on=group_cols, how="left")

# Consistency: std dev of weekly points (lower = more consistent)
std = df.groupby(group_cols, as_index=False)[pts_col].std()
std.rename(columns={pts_col: "weekly_std"}, inplace=True)
std["weekly_std"] = std["weekly_std"].round(1)
agg = agg.merge(std, on=group_cols, how="left")

# Boom weeks (20+ pts) and bust weeks (<5 pts)
boom = df[df[pts_col] >= 20].groupby(group_cols, as_index=False)[pts_col].count()
boom.rename(columns={pts_col: "boom_weeks"}, inplace=True)
bust = df[df[pts_col] < 5].groupby(group_cols, as_index=False)[pts_col].count()
bust.rename(columns={pts_col: "bust_weeks"}, inplace=True)
agg = agg.merge(boom, on=group_cols, how="left").merge(bust, on=group_cols, how="left")
agg["boom_weeks"] = agg["boom_weeks"].fillna(0).astype(int)
agg["bust_weeks"] = agg["bust_weeks"].fillna(0).astype(int)

# Sort and rank
agg = agg.sort_values(pts_col, ascending=False).head(top_n).reset_index(drop=True)
agg.insert(0, "Rank", range(1, len(agg) + 1))

# ── Top 3 cards ──────────────────────────────────────────────────────────────
scoring_label = scoring
st.markdown(f"### {sel_season} Fantasy Leaders ({scoring_label})")

if len(agg) >= 3:
    c1, c2, c3 = st.columns(3)
    for i, (col, medal) in enumerate(zip([c1, c2, c3], ["🥇 #1", "🥈 #2", "🥉 #3"])):
        row = agg.iloc[i]
        player = row[name_col]
        team = row[team_col] if team_col else "—"
        pos  = row[pos_col] if pos_col else ""
        pts  = row[pts_col]
        ppg  = row["ppg"]
        logo_html = ""
        url = get_logo(team, teams)
        if url:
            logo_html = f'<img src="{url}" width="40" style="margin:6px 0;">'
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">{medal}</div>
                {logo_html}
                <div class="value" style="font-size:1.15rem;">{player}</div>
                <div class="sub">{team} · {pos} &nbsp;·&nbsp; {pts:,.1f} pts &nbsp;·&nbsp; {ppg} ppg</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Summary metrics ──────────────────────────────────────────────────────────
if len(agg) >= 1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg PPG (Top N)", f"{agg['ppg'].mean():.1f}")
    m2.metric("Avg Total Pts", f"{agg[pts_col].mean():,.0f}")
    m3.metric("Most Boom Weeks", f"{agg['boom_weeks'].max()}")
    m4.metric("Most TDs", f"{agg['total_tds'].max()}")

st.markdown("---")

# ── Bar chart: top scorers ───────────────────────────────────────────────────
st.markdown(f"#### Top {top_n} Fantasy Scorers")

bar_colors = [TEAM_COLORS.get(row.get(team_col, ""), "#4f46e5") for _, row in agg.iterrows()]

fig = px.bar(
    agg, x=name_col, y=pts_col,
    hover_data=["ppg", "games", "total_tds"],
)
fig.update_traces(marker_color=bar_colors)

y_max = agg[pts_col].max()
logo_h = y_max * 0.13
fig.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{sel_season} Fantasy Points ({scoring_label})",
    xaxis_title="Player", yaxis_title="Fantasy Points",
    showlegend=False,
    xaxis_tickangle=-35,
    yaxis_range=[0, y_max * 1.22],
)
if team_col:
    for idx, (_, row) in enumerate(agg.iterrows()):
        url = get_logo(row[team_col], teams)
        if url:
            fig.add_layout_image(dict(
                source=url, xref="x", yref="y",
                x=idx, y=row[pts_col] + y_max * 0.01,
                sizex=0.8, sizey=logo_h,
                xanchor="center", yanchor="bottom", layer="above",
            ))
st.plotly_chart(fig, use_container_width=True)

# ── Points per game chart ────────────────────────────────────────────────────
st.markdown("#### Points Per Game")
ppg_df = agg[agg["games"] >= 4].sort_values("ppg", ascending=False).head(top_n)

if not ppg_df.empty:
    ppg_colors = [TEAM_COLORS.get(row.get(team_col, ""), "#10b981") for _, row in ppg_df.iterrows()]
    fig_ppg = px.bar(
        ppg_df, x=name_col, y="ppg",
        hover_data=[pts_col, "games", "total_tds"],
    )
    fig_ppg.update_traces(marker_color=ppg_colors)

    ppg_max = ppg_df["ppg"].max()
    ppg_logo_h = ppg_max * 0.13
    fig_ppg.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Fantasy PPG ({scoring_label}, min 4 games)",
        xaxis_title="Player", yaxis_title="Points Per Game",
        showlegend=False,
        xaxis_tickangle=-35,
        yaxis_range=[0, ppg_max * 1.22],
    )
    if team_col:
        for idx, (_, row) in enumerate(ppg_df.iterrows()):
            url = get_logo(row[team_col], teams)
            if url:
                fig_ppg.add_layout_image(dict(
                    source=url, xref="x", yref="y",
                    x=idx, y=row["ppg"] + ppg_max * 0.01,
                    sizex=0.8, sizey=ppg_logo_h,
                    xanchor="center", yanchor="bottom", layer="above",
                ))
    st.plotly_chart(fig_ppg, use_container_width=True)

st.markdown("---")

# ── Weekly consistency: top 5 players line chart ─────────────────────────────
st.markdown("#### Weekly Scoring Trend")
top5 = agg.head(5)[name_col].tolist()
trend_df = df[df[name_col].isin(top5)].copy()

if not trend_df.empty and "week" in trend_df.columns:
    trend_df = trend_df.sort_values("week")
    fig_trend = go.Figure()
    for player in top5:
        p_df = trend_df[trend_df[name_col] == player]
        team_abbr = p_df[team_col].iloc[0] if team_col and not p_df.empty else ""
        color = TEAM_COLORS.get(team_abbr, "#4f46e5")
        fig_trend.add_trace(go.Scatter(
            x=p_df["week"], y=p_df[pts_col].round(1),
            mode="lines+markers",
            name=player,
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color, line=dict(color="#fff", width=1)),
            hovertemplate=f"<b>{player}</b><br>Week %{{x}}<br>{scoring_label}: %{{y:.1f}}<extra></extra>",
        ))
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Weekly Fantasy Points — Top 5 ({scoring_label})",
        xaxis_title="Week", yaxis_title="Fantasy Points",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ── Full leaderboard table ───────────────────────────────────────────────────
st.markdown("#### Full Fantasy Leaderboard")

table_cols = ["Rank", name_col]
if team_col: table_cols.append(team_col)
if pos_col:  table_cols.append(pos_col)
table_cols += [pts_col, "ppg", "games", "total_tds"]
if "boom_weeks" in agg.columns:
    table_cols.append("boom_weeks")
if "bust_weeks" in agg.columns:
    table_cols.append("bust_weeks")
if "weekly_std" in agg.columns:
    table_cols.append("weekly_std")

# Only include columns that exist
table_cols = [c for c in table_cols if c in agg.columns]

rename_map = {
    name_col: "Player",
    pts_col: "Total Pts",
    "ppg": "PPG",
    "games": "GP",
    "total_tds": "TDs",
    "boom_weeks": "Boom",
    "bust_weeks": "Bust",
    "weekly_std": "Std Dev",
}
if team_col: rename_map[team_col] = "Team"
if pos_col:  rename_map[pos_col] = "Pos"

# Add position-specific stat columns
if sel_pos == "QB" or sel_pos == "All":
    for c in ["passing_yards", "passing_tds", "interceptions"]:
        if c in agg.columns:
            table_cols.append(c)
            rename_map[c] = {"passing_yards": "Pass Yds", "passing_tds": "Pass TD", "interceptions": "INT"}[c]

if sel_pos in ("RB", "All"):
    for c in ["rushing_yards", "rushing_tds", "carries"]:
        if c in agg.columns and c not in table_cols:
            table_cols.append(c)
            rename_map[c] = {"rushing_yards": "Rush Yds", "rushing_tds": "Rush TD", "carries": "Carries"}[c]

if sel_pos in ("WR", "TE", "All"):
    for c in ["receptions", "targets", "receiving_yards", "receiving_tds"]:
        if c in agg.columns and c not in table_cols:
            table_cols.append(c)
            rename_map[c] = {"receptions": "Rec", "targets": "Tgt", "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD"}[c]

# Round float columns for display
disp = agg[table_cols].copy()
for c in disp.select_dtypes("float").columns:
    disp[c] = disp[c].round(2 if c == pts_col else 1)

st.dataframe(
    disp.rename(columns=rename_map),
    hide_index=True,
    use_container_width=True,
    column_config={"Total Pts": st.column_config.NumberColumn(format="%.2f")},
)

st.markdown("---")

# ── Preseason Ranking vs Actual Finish ───────────────────────────────────────
st.markdown("#### Preseason ESPN Ranking vs Actual Finish")

pre_df = load_preseason_rankings()

if pre_df.empty:
    st.info(
        "Preseason rankings not loaded yet. Run the fetcher first:\n\n"
        "```\npython src/load_preseason_rankings.py\n```"
    )
else:
    season_pre = pre_df[pre_df["season"] == sel_season].copy()
    if season_pre.empty:
        st.info(f"No preseason rankings available for {sel_season}.")
    else:
        # Build actual finish ranks across all positions for the whole season
        if name_col:
            actual_all = weekly[weekly["season"] == sel_season].copy()
            if pos_col:
                actual_all = actual_all[actual_all[pos_col].isin(FANTASY_POSITIONS)]
            if pts_col not in actual_all.columns:
                actual_all[pts_col] = actual_all.get("fantasy_points", 0)

            grp_cols = [name_col] + ([team_col] if team_col else []) + ([pos_col] if pos_col else [])
            actual_agg = actual_all.groupby(grp_cols, as_index=False)[pts_col].sum()
            actual_agg = actual_agg.sort_values(pts_col, ascending=False).reset_index(drop=True)
            actual_agg["actual_rank"] = range(1, len(actual_agg) + 1)
            actual_agg["name_key"] = actual_agg[name_col].apply(_normalize_name)

            # Normalize preseason name key for joining
            if "name_key" not in season_pre.columns:
                season_pre["name_key"] = season_pre["player_name"].apply(_normalize_name)

            # Bring position from actual data; rename to avoid collision with
            # the preseason CSV's own (mostly-empty) "position" column.
            merge_actual_cols = ["name_key", "actual_rank", pts_col]
            if pos_col:
                actual_agg = actual_agg.rename(columns={pos_col: "actual_pos"})
                merge_actual_cols.append("actual_pos")

            merged = season_pre.drop(columns=["position"], errors="ignore").merge(
                actual_agg[merge_actual_cols],
                on="name_key",
                how="inner",
            )

            if merged.empty:
                st.warning("Could not match preseason ranking names to player data.")
            else:
                merged["net_score"] = merged["preseason_rank"] - merged["actual_rank"]
                merged = merged.sort_values("preseason_rank").reset_index(drop=True)

                pos_display_col = "actual_pos" if "actual_pos" in merged.columns else None
                show_cols = ["player_name"] + ([pos_display_col] if pos_display_col else []) + \
                            ["preseason_rank", "actual_rank", "net_score", pts_col]
                show = merged[show_cols].copy()
                show[pts_col] = show[pts_col].round(2)
                rename_show = {"player_name": "Player", "preseason_rank": "Preseason Rank",
                               "actual_rank": "Actual Rank", "net_score": "Net Score",
                               pts_col: "Total Pts"}
                if pos_display_col:
                    rename_show[pos_display_col] = "Pos"
                show = show.rename(columns=rename_show)

                def _color_net(val):
                    if val > 0:
                        intensity = min(int(abs(val) * 4), 180)
                        return f"color: rgb(5,{100+intensity//2},5); font-weight:600"
                    elif val < 0:
                        intensity = min(int(abs(val) * 4), 180)
                        return f"color: rgb({100+intensity//2},5,5); font-weight:600"
                    return ""

                try:
                    styled = show.style.map(_color_net, subset=["Net Score"])
                except AttributeError:
                    styled = show.style.applymap(_color_net, subset=["Net Score"])
                st.caption(
                    "Net Score = Preseason Rank − Actual Rank  ·  "
                    "**Positive (green)** = outperformed expectations  ·  "
                    "**Negative (red)** = underperformed"
                )
                st.dataframe(
                    styled, hide_index=True, use_container_width=True,
                    column_config={"Total Pts": st.column_config.NumberColumn(format="%.2f")},
                )
