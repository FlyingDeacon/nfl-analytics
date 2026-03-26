import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import (
    load_ratings, load_teams, load_schedules, load_weekly,
    get_logo, add_ranks, load_depth_charts, load_divisions,
    get_base_dir, _file_mtime,
)
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Team Profile · NFL", page_icon="🏟️", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)
render_sidebar_nav(current_page="8_Team_Profile")

# ── Back button ──────────────────────────────────────────────────────────────
if st.button("← Back to Team Ratings", key="back_to_ratings"):
    st.switch_page("pages/1_Team_Ratings.py")

# ── Load data (pass mtime so cache auto-invalidates when files change) ────────
_base = get_base_dir()
ratings      = load_ratings(_mtime=_file_mtime(_base / "data/processed/team_ratings.csv"))
teams_df     = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))
schedules    = load_schedules(_mtime=_file_mtime(_base / "data/raw/schedules.csv"))
weekly       = load_weekly(_mtime=_file_mtime(_base / "data/raw/weekly.csv"))
depth_charts = load_depth_charts(_mtime=_file_mtime(_base / "data/raw/depth_charts.csv"))
divisions_df = load_divisions(_mtime=_file_mtime(_base / "data/raw/nfl_divisions.csv"))

# ── Sidebar: season + single team selector ───────────────────────────────────
seasons = sorted(ratings["season"].dropna().unique().astype(int), reverse=True)
sel_season = st.sidebar.selectbox("Season", seasons, key="tp_season")

team_list = sorted(ratings[ratings["season"] == sel_season]["team"].unique().tolist())

# Honour pre-selection passed from Team Ratings (via session_state)
default_team = st.session_state.get("profile_team", team_list[0])
if default_team not in team_list:
    default_team = team_list[0]

sel_team = st.sidebar.selectbox(
    "Select Team",
    team_list,
    index=team_list.index(default_team),
    key="tp_team",
)

# Inline team picker for direct page filter
page_sel_team = st.selectbox(
    "Quick Select Team",
    team_list,
    index=team_list.index(sel_team),
    key="tp_team_page",
    help="Use this dropdown on the page to switch team profiles without the sidebar.",
)

# Keep profile team in session state; selection is immediate so no explicit rerun needed.
sel_team = page_sel_team
st.session_state["profile_team"] = sel_team

# ── Team metadata ────────────────────────────────────────────────────────────
abbr_col  = "team_abbr" if "team_abbr" in teams_df.columns else "team"
team_row  = teams_df[teams_df[abbr_col] == sel_team]
team_name = team_row.iloc[0]["team_name"]      if not team_row.empty else sel_team
team_div  = team_row.iloc[0]["team_division"]  if not team_row.empty else ""
logo_url  = get_logo(sel_team, teams_df)
team_color = TEAM_COLORS.get(sel_team, "#4f46e5")

# ── Page header ──────────────────────────────────────────────────────────────
logo_html = f'<img src="{logo_url}" width="72" style="margin-right:18px;object-fit:contain;">' if logo_url else ""
st.markdown(f"""
<div class="nfl-page-header" style="align-items:center;">
    {logo_html}
    <div>
        <div class="title">{team_name}</div>
        <div class="subtitle">{team_div} &nbsp;·&nbsp; {sel_season} Season Profile</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Team ratings for selected season ─────────────────────────────────────────
season_ratings = add_ranks(ratings[ratings["season"] == sel_season].copy())
team_stats = season_ratings[season_ratings["team"] == sel_team]

if team_stats.empty:
    st.warning(f"No rating data found for {sel_team} in {sel_season}.")
    st.stop()

row = team_stats.iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# STAT CARDS — PPG / OPPG / Net Rating with ranks
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns(3)
card_defs = [
    (c1, "Offensive PPG",  f"{row['ppg']:.1f}",      f"Rank #{int(row['offense_rank'])} of 32",  "⚔️"),
    (c2, "Defensive PPG",  f"{row['oppg']:.1f}",     f"Rank #{int(row['defense_rank'])} of 32",  "🛡️"),
    (c3, "Net Rating",     f"{row['net_ppg']:+.1f}",  f"Rank #{int(row['overall_rank'])} of 32", "📈"),
]
for col, label, val, rank_str, icon in card_defs:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:20px 12px;">
            <div class="label">{icon} {label}</div>
            <div class="value" style="font-size:2.2rem;font-weight:800;color:{team_color};margin:8px 0 4px;">{val}</div>
            <div class="sub" style="font-size:0.9rem;">{rank_str}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RECORD + DIVISION STANDING
# ══════════════════════════════════════════════════════════════════════════════

reg_games = schedules[
    (schedules["season"] == sel_season) &
    (schedules["game_type"] == "REG")
].copy()
reg_games["home_score"] = pd.to_numeric(reg_games.get("home_score", pd.Series(dtype=float)), errors="coerce")
reg_games["away_score"] = pd.to_numeric(reg_games.get("away_score", pd.Series(dtype=float)), errors="coerce")

def calc_record(df, team):
    """Return (W, L, T, PF, PA) for a team from a games dataframe."""
    home = df[df["home_team"] == team][["home_score", "away_score"]].rename(
        columns={"home_score": "pf", "away_score": "pa"}
    )
    away = df[df["away_team"] == team][["away_score", "home_score"]].rename(
        columns={"away_score": "pf", "home_score": "pa"}
    )
    both = pd.concat([home, away]).dropna()
    if both.empty:
        return 0, 0, 0, 0, 0
    w  = int((both["pf"] > both["pa"]).sum())
    l  = int((both["pf"] < both["pa"]).sum())
    t  = int((both["pf"] == both["pa"]).sum())
    pf = int(both["pf"].sum())
    pa = int(both["pa"].sum())
    return w, l, t, pf, pa


def win_pct(w, l, t):
    total = w + l + t
    return (w + 0.5 * t) / max(total, 1)


def _h2h_pct(games_df, team, opponents):
    """Win % for `team` in games against a specific set of `opponents`."""
    h2h = games_df[
        ((games_df["home_team"] == team) & (games_df["away_team"].isin(opponents))) |
        ((games_df["away_team"] == team) & (games_df["home_team"].isin(opponents)))
    ]
    w, l, t, _, _ = calc_record(h2h, team)
    return win_pct(w, l, t)


def _div_pct(games_df, team, div_teams):
    """Win % for `team` in division games."""
    opp = [x for x in div_teams if x != team]
    div_g = games_df[
        ((games_df["home_team"] == team) & (games_df["away_team"].isin(opp))) |
        ((games_df["away_team"] == team) & (games_df["home_team"].isin(opp)))
    ]
    w, l, t, _, _ = calc_record(div_g, team)
    return win_pct(w, l, t)


def _conf_pct(games_df, team, conf_teams):
    """Win % for `team` in conference games."""
    conf_g = games_df[
        ((games_df["home_team"] == team) & (games_df["away_team"].isin(conf_teams))) |
        ((games_df["away_team"] == team) & (games_df["home_team"].isin(conf_teams)))
    ]
    w, l, t, _, _ = calc_record(conf_g, team)
    return win_pct(w, l, t)


def nfl_tiebreak_sort(teams, games_df, div_teams, conf_teams):
    """
    Sort teams using NFL tiebreaker rules (simplified):
      1. Overall win %
      2. H2H win % within the tied group
      3. Division win %
      4. Conference win %
      5. Points For (season total)
    Returns list of teams in correct standing order.
    """
    if len(teams) <= 1:
        return teams

    def sort_key(team):
        w, l, t, pf, _ = calc_record(games_df, team)
        overall  = win_pct(w, l, t)
        h2h      = _h2h_pct(games_df, team, [x for x in teams if x != team])
        div      = _div_pct(games_df, team, div_teams)
        conf     = _conf_pct(games_df, team, conf_teams)
        return (overall, h2h, div, conf, pf)

    return sorted(teams, key=sort_key, reverse=True)


# ── Division setup ────────────────────────────────────────────────────────────
# Use nfl_divisions.csv as authoritative source (canonical 4 teams per division)
if team_div and not divisions_df.empty:
    div_teams = divisions_df.loc[
        divisions_df["division"] == team_div, "team_abbr"
    ].dropna().unique().tolist()
else:
    div_teams = teams_df.loc[
        teams_df["team_division"] == team_div, abbr_col
    ].dropna().unique().tolist() if team_div else []

if sel_team not in div_teams:
    div_teams.append(sel_team)

# Conference teams (for tiebreaker step 4)
conf = "AFC" if team_div.startswith("AFC") else "NFC"
if not divisions_df.empty:
    conf_teams = divisions_df.loc[
        divisions_df["conference"] == conf, "team_abbr"
    ].dropna().unique().tolist()
else:
    conf_teams = div_teams  # fallback

div_opponents = [t for t in div_teams if t != sel_team]

# ── Records ───────────────────────────────────────────────────────────────────
ow, ol, ot, _, _ = calc_record(reg_games, sel_team)
overall_rec = f"{ow}–{ol}" + (f"–{ot}" if ot else "")

div_g = reg_games[
    ((reg_games["home_team"] == sel_team) & (reg_games["away_team"].isin(div_opponents))) |
    ((reg_games["away_team"] == sel_team) & (reg_games["home_team"].isin(div_opponents)))
]
dw, dl, dt, _, _ = calc_record(div_g, sel_team)
div_rec = f"{dw}–{dl}" + (f"–{dt}" if dt else "")

# ── Division standings with NFL tiebreakers ───────────────────────────────────
div_standing_rows = []
for t in div_teams:
    tw, tl, tt, tpf, tpa = calc_record(reg_games, t)
    total_g = tw + tl + tt
    div_standing_rows.append({
        "team": t,
        "W": tw, "L": tl, "T": tt,
        "win_pct": round(win_pct(tw, tl, tt), 3),
        "_pf": tpf, "_pa": tpa,
    })

raw_df = pd.DataFrame(div_standing_rows)

# Apply tiebreakers within groups that share the same win %
sorted_teams = []
for _, grp in raw_df.groupby("win_pct", sort=False):
    if len(grp) == 1:
        sorted_teams.append(grp["team"].iloc[0])
    else:
        # Multiple teams tied — apply NFL tiebreaker order
        tied = grp["team"].tolist()
        sorted_teams.extend(nfl_tiebreak_sort(tied, reg_games, div_teams, conf_teams))

# Rebuild dataframe in tiebreaker-resolved order
sorted_rows = [raw_df[raw_df["team"] == t].iloc[0].to_dict() for t in sorted_teams]
div_standing_df = pd.DataFrame(sorted_rows)

# Use Categorical so sort_values respects tiebreaker order within same win_pct
div_standing_df["team"] = pd.Categorical(
    div_standing_df["team"], categories=sorted_teams, ordered=True
)
div_standing_df = div_standing_df.sort_values(
    ["win_pct", "team"], ascending=[False, True]
).reset_index(drop=True)
div_standing_df.insert(0, "Place", range(1, len(div_standing_df) + 1))

place_row  = div_standing_df[div_standing_df["team"] == sel_team]
team_place = int(place_row.iloc[0]["Place"]) if not place_row.empty else 0
ordinals   = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
place_str  = ordinals.get(team_place, f"{team_place}th")

st.markdown("### 📋 Season Record & Division Standing")
r1, r2, r3 = st.columns(3)
record_cards = [
    (r1, "Overall Record",   overall_rec, f"{int(row['games'])} games played"),
    (r2, "Division Record",  div_rec,     team_div),
    (r3, "Division Standing", place_str,  team_div),
]
for col, label, val, sub in record_cards:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:20px 12px;">
            <div class="label">{label}</div>
            <div class="value" style="font-size:2.2rem;font-weight:800;color:{team_color};margin:8px 0 4px;">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Division standings table
st.markdown(f"#### {team_div} Standings — {sel_season}")

def _logo_cell(abbr):
    url = get_logo(abbr, teams_df)
    return f'<img src="{url}" width="26" style="vertical-align:middle;margin-right:6px;">' if url else ""

standings_html = """
<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.92rem;">
<thead>
  <tr style="border-bottom:2px solid #e2e5ef;color:#4a4e69;text-align:left;">
    <th style="padding:8px 6px;">Place</th>
    <th style="padding:8px 6px;">Team</th>
    <th style="padding:8px 6px;text-align:center;">W</th>
    <th style="padding:8px 6px;text-align:center;">L</th>
    <th style="padding:8px 6px;text-align:center;">T</th>
    <th style="padding:8px 6px;text-align:center;">Win %</th>
  </tr>
</thead>
<tbody>
"""
for _, srow in div_standing_df.iterrows():
    t       = srow["team"]
    is_sel  = t == sel_team
    bg      = f"background:{team_color}18;" if is_sel else ""
    fw      = "font-weight:700;" if is_sel else ""
    logo    = _logo_cell(t)
    rec_str = f"{int(srow['W'])}–{int(srow['L'])}" + (f"–{int(srow['T'])}" if srow['T'] else "")
    standings_html += f"""
  <tr style="{bg}{fw}border-bottom:1px solid #e8eaef;">
    <td style="padding:8px 6px;">{ordinals.get(int(srow['Place']), str(int(srow['Place'])))}</td>
    <td style="padding:8px 6px;">{logo}{t}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['W'])}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['L'])}</td>
    <td style="padding:8px 6px;text-align:center;">{int(srow['T'])}</td>
    <td style="padding:8px 6px;text-align:center;">{srow['win_pct']:.3f}</td>
  </tr>"""
standings_html += "</tbody></table>"
st.markdown(standings_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DEPTH CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Build detailed stats from weekly for merging into depth chart ─────────────
# Depth chart already has GP + fantasy_pts from the rebuild.
# We only need per-stat breakdowns (pass yds, rush yds, targets, etc.) from weekly.
_wk = weekly.copy() if not weekly.empty else pd.DataFrame()
if not _wk.empty:
    if "season_type" in _wk.columns:
        _wk = _wk[_wk["season_type"] == "REG"]
    _wk_team = _wk[
        (_wk["recent_team"] == sel_team) & (_wk["season"] == sel_season)
    ].copy()
    detail_cols = [
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    ]
    detail_cols = [c for c in detail_cols if c in _wk_team.columns]
    if not _wk_team.empty and detail_cols:
        _stats = _wk_team.groupby("player_display_name", as_index=False)[detail_cols].sum()
    else:
        _stats = pd.DataFrame()
else:
    _wk_team = pd.DataFrame()
    _stats   = pd.DataFrame()


def _merge_stats(dc_pos_df):
    """Merge depth chart rows with per-stat breakdowns from weekly.
    GP, PPR Pts and PPG come from the depth chart rebuild — not re-computed here.
    """
    player_col = "Player" if "Player" in dc_pos_df.columns else "player_name"

    # Rename depth chart fantasy_pts → PPR Pts, compute PPG
    out = dc_pos_df.copy()
    if "fantasy_pts" in out.columns:
        out = out.rename(columns={"fantasy_pts": "PPR Pts"})
    if "GP" in out.columns and "PPR Pts" in out.columns:
        out["PPG"] = (
            pd.to_numeric(out["PPR Pts"], errors="coerce") /
            out["GP"].clip(lower=1)
        ).round(1)

    if _stats.empty:
        return out

    merged = out.merge(
        _stats,
        left_on=player_col,
        right_on="player_display_name",
        how="left",
    ).drop(columns=["player_display_name"], errors="ignore")

    return merged


# Get team's depth chart rows from nflverse data
_dc = depth_charts[(depth_charts["team"] == sel_team)].copy() if not depth_charts.empty else pd.DataFrame()

# preserve season label for caption
if not _dc.empty and "season" in _dc.columns:
    dc_season = int(_dc["season"].mode()[0])
else:
    dc_season = sel_season

# ══════════════════════════════════════════════════════════════════════════════
# OFFENSIVE DEPTH CHART  (sourced from actual 2025 REG game logs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏈 Offensive Depth Chart")
st.caption("2025 regular season · ranked by games played & stats · all 32 teams")

OFF_POS_GROUPS = [
    ("QB", "Quarterbacks"),
    ("RB", "Running Backs"),
    ("WR", "Wide Receivers"),
    ("TE", "Tight Ends"),
]

OFF_STAT_COLS = {
    "QB": {
        "cols":   ["Player", "GP", "PPR Pts", "PPG",
                   "completions", "attempts", "passing_yards", "passing_tds",
                   "interceptions", "carries", "rushing_yards", "rushing_tds"],
        "rename": {
            "completions": "CMP", "attempts": "ATT",
            "passing_yards": "Pass Yds", "passing_tds": "Pass TD",
            "interceptions": "INT", "carries": "Rush Att",
            "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
        },
    },
    "RB": {
        "cols":   ["Player", "GP", "PPR Pts", "PPG",
                   "carries", "rushing_yards", "rushing_tds",
                   "receptions", "targets", "receiving_yards", "receiving_tds"],
        "rename": {
            "carries": "Car", "rushing_yards": "Rush Yds", "rushing_tds": "Rush TD",
            "receptions": "Rec", "targets": "Tgt",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
        },
    },
    "WR": {
        "cols":   ["Player", "GP", "PPR Pts", "PPG",
                   "targets", "receptions", "receiving_yards", "receiving_tds",
                   "carries", "rushing_yards"],
        "rename": {
            "targets": "Tgt", "receptions": "Rec",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
            "carries": "Rush Att", "rushing_yards": "Rush Yds",
        },
    },
    "TE": {
        "cols":   ["Player", "GP", "PPR Pts", "PPG",
                   "targets", "receptions", "receiving_yards", "receiving_tds"],
        "rename": {
            "targets": "Tgt", "receptions": "Rec",
            "receiving_yards": "Rec Yds", "receiving_tds": "Rec TD",
        },
    },
}

if _dc.empty:
    st.info("Depth chart data not available for this team.")
else:
    dc_off = _dc[_dc["side"] == "offense"].copy()

    for pos, label in OFF_POS_GROUPS:
        pos_df = dc_off[dc_off["position"] == pos].sort_values("depth_order").copy()
        if pos_df.empty:
            continue

        pos_df = pos_df.rename(columns={"player_name": "Player"})

        # Merge 2025 season stats from weekly.csv
        merged = _merge_stats(pos_df)

        cfg  = OFF_STAT_COLS.get(pos, {"cols": ["Player", "GP"], "rename": {}})
        show = [c for c in cfg["cols"] if c in merged.columns]

        # Round stat columns; keep PPG/PPR Pts as formatted strings
        for fc in merged.select_dtypes("float").columns:
            if fc in ("PPG", "PPR Pts"):
                pass  # handled below
            else:
                merged[fc] = merged[fc].fillna(0).round(0).astype(int, errors="ignore")
        if "PPG"     in merged.columns:
            merged["PPG"]     = merged["PPG"].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x != "—" else "—")
        if "PPR Pts" in merged.columns:
            merged["PPR Pts"] = merged["PPR Pts"].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) and x != "—" else "—")

        disp = merged[show].rename(columns=cfg["rename"])

        st.markdown(f"**{label}**")
        st.dataframe(
            disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PPG":     st.column_config.TextColumn("PPG"),
                "PPR Pts": st.column_config.TextColumn("PPR Pts"),
                "GP":      st.column_config.NumberColumn("GP", width="small"),
            },
        )
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DEFENSIVE DEPTH CHART  (sourced from actual 2025 REG game logs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🛡️ Defensive Depth Chart")
st.caption("2025 regular season · ranked by games played")

DEF_POS_GROUPS = [
    ("Defensive Line",  ["DL"]),
    ("Defensive Backs", ["DB"]),
]

if _dc.empty:
    st.info("Depth chart data not available for this team.")
else:
    dc_def = _dc[_dc["side"] == "defense"].copy()

    if dc_def.empty:
        st.info("No defensive data available for this team.")
    else:
        for group_label, positions in DEF_POS_GROUPS:
            g_df = dc_def[dc_def["position"].isin(positions)].sort_values("depth_order").copy()
            if g_df.empty:
                continue

            show_cols = ["player_name", "GP"] if "GP" in g_df.columns else ["player_name"]
            disp = g_df[show_cols].rename(columns={"player_name": "Player"})

            st.markdown(f"**{group_label}**")
            st.dataframe(
                disp,
                hide_index=True,
                use_container_width=True,
                column_config={"GP": st.column_config.NumberColumn("GP", width="small")},
            )
            st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    f"**Data sources** — Season ratings from processed team stats · "
    f"Records calculated from {sel_season} regular season schedule · "
    "Depth chart based on projected 2026 rosters · "
    "Season stats from 2025 regular season game logs."
)
