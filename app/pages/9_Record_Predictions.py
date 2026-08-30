import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_teams, get_logo, get_base_dir, _file_mtime
from utils.projection import season_projection
from utils.nav import render_sidebar_nav
from utils.gate import require_passcode

st.set_page_config(page_title="Season Projections · NFL", page_icon="🔮", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# Gate before anything else: this page and the two it leads to are the CHOPPED
# edge, which is only worth having if the rest of the pool cannot read it.
require_passcode("2026 Season Projections")

render_sidebar_nav(current_page="9_Record_Predictions")

if st.button("← Back to Team Ratings", key="back_to_ratings"):
    st.switch_page("pages/1_Team_Ratings.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔮</div>
    <div>
        <div class="title">2026 Season Projections</div>
        <div class="subtitle">Projected record & expected finish · powered by 2025 results + updated rosters</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Load data + run projection (shared cache with the matchup/survivor pages) ─
_base = get_base_dir()
teams_df = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))

table, games, changes = season_projection()
_market = table.attrs.get("market", {})

# ── Companion pages ──────────────────────────────────────────────────────────
_nav_l, _nav_r = st.columns(2)
if _nav_l.button("🗓️ Weekly Matchups", key="rp_to_matchups", use_container_width=True,
                 help="Every game this week with win probabilities and Vegas lines"):
    st.switch_page("pages/12_Weekly_Matchups.py")
if _nav_r.button("🔪 CHOPPED Survivor", key="rp_to_chopped", use_container_width=True,
                 help="Which team to pick each week, and what each pick costs later"):
    st.switch_page("pages/13_Chopped_Survivor.py")

ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def _rec(w):
    wi = int(round(w))
    return f"{wi}–{17 - wi}"


# ══════════════════════════════════════════════════════════════════════════════
# LEAGUE PROJECTIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏆 League Projections")

conf_opt = st.radio("Conference", ["All", "AFC", "NFC"], horizontal=True, key="rp_conf")
league = table if conf_opt == "All" else table[table["conference"] == conf_opt]
league = league.reset_index(drop=True)

disp = pd.DataFrame({
    "Logo": [get_logo(t, teams_df) for t in league["team"]],
    "Team": league["team"],
    "Division": league["division"],
    "Proj Record": [_rec(w) for w in league["proj_wins"]],
    "Proj Wins": league["proj_wins"].round(1),
    "Proj Off PPG": league["proj_off_ppg"].round(1),
    "Off Δ vs '25": league["off_change"].round(1),
    "Proj Def PPG": league["proj_def_ppg"].round(1),
    "Def Δ vs '25": league["def_change"].round(1),
    "Power": league["power"].round(1),
    "Make Playoffs": league["playoff_pct"].round(0),
    "Win Division": league["div_title_pct"].round(0),
    "Proj Finish": league["exp_finish"].round(2),
})
st.dataframe(
    disp,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Logo": st.column_config.ImageColumn("", width="small"),
        "Proj Wins": st.column_config.NumberColumn("Proj Wins", format="%.1f"),
        "Proj Off PPG": st.column_config.NumberColumn("Proj Off PPG", format="%.1f",
                    help="Offense projected from the 2026 roster (calibrated to 2025)"),
        "Off Δ vs '25": st.column_config.NumberColumn("Off Δ vs '25", format="%+.1f",
                    help="Predicted offensive PPG shift from offseason roster changes"),
        "Proj Def PPG": st.column_config.NumberColumn("Proj Def PPG", format="%.1f",
                    help="Projected points allowed per game (2025 results adjusted for roster turnover; lower = better)"),
        "Def Δ vs '25": st.column_config.NumberColumn("Def Δ vs '25", format="%+.1f",
                    help="Points-allowed shift from defensive roster changes (negative = improved defense)"),
        "Power": st.column_config.NumberColumn("Power", format="%+.1f",
                    help="Projected point margin vs a league-average team"),
        "Make Playoffs": st.column_config.NumberColumn("Playoffs %", format="%d%%"),
        "Win Division": st.column_config.NumberColumn("Div Title %", format="%d%%"),
        "Proj Finish": st.column_config.NumberColumn("Avg Div Finish", format="%.2f",
                    help="Average finishing place within the division across simulations"),
    },
)
_anchor = (f"power blended {int(_market.get('weight', 0)*100)}% to market win totals · "
           if _market.get("used") else "")
st.caption(
    f"Sorted by projected wins · 20,000 simulated seasons · {_anchor}"
    f"model rating = 2025 points scored & allowed, regressed toward the mean and "
    f"adjusted for offseason roster turnover"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PROJECTED DIVISION STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Projected Division Standings")

div_order = ["AFC East", "AFC North", "AFC South", "AFC West",
             "NFC East", "NFC North", "NFC South", "NFC West"]
grid = st.columns(2)
for i, dv in enumerate(div_order):
    d = table[table["division"] == dv].sort_values("proj_wins", ascending=False).reset_index(drop=True)
    if d.empty:
        continue
    html = f'<div style="margin-bottom:6px;"><b style="color:#4a4e69;">{dv}</b></div>'
    html += '<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.9rem;margin-bottom:18px;">'
    for place, (_, r) in enumerate(d.iterrows(), start=1):
        url = get_logo(r["team"], teams_df)
        logo = f'<img src="{url}" width="22" style="vertical-align:middle;margin-right:6px;">' if url else ""
        lead = "font-weight:700;" if place == 1 else ""
        html += (
            f'<tr style="border-bottom:1px solid #eceef4;{lead}">'
            f'<td style="padding:6px 4px;color:#8b8fa8;width:28px;">{ORD[place]}</td>'
            f'<td style="padding:6px 4px;">{logo}{r["team"]}</td>'
            f'<td style="padding:6px 4px;text-align:right;">{_rec(r["proj_wins"])}</td>'
            f'<td style="padding:6px 4px;text-align:right;color:#4f46e5;">{r["playoff_pct"]:.0f}% PO</td>'
            f'</tr>'
        )
    html += "</table>"
    grid[i % 2].markdown(html, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TEAM FOCUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 Team Deep Dive")

team_names = table.sort_values("team")["team"].tolist()
default = st.session_state.get("profile_team", team_names[0])
sel = st.selectbox("Select a team", team_names,
                   index=team_names.index(default) if default in team_names else 0,
                   key="rp_team")
trow = table[table["team"] == sel].iloc[0]
tcolor = TEAM_COLORS.get(sel, "#4f46e5")

m1, m2, m3, m4 = st.columns(4)
cards = [
    (m1, "Projected Record", _rec(trow["proj_wins"]), f"{trow['proj_wins']:.1f} expected wins"),
    (m2, "Power Rating", f"{trow['power']:+.1f}", "pts vs avg team"),
    (m3, "Make Playoffs", f"{trow['playoff_pct']:.0f}%", "across 20k sims"),
    (m4, "Win Division", f"{trow['div_title_pct']:.0f}%", f"proj finish {ORD.get(int(round(trow['exp_finish'])), '—')}"),
]
for col, label, val, sub in cards:
    with col:
        st.markdown(f"""
        <div class="stat-card" style="text-align:center;padding:18px 12px;">
            <div class="label">{label}</div>
            <div class="value" style="font-size:2rem;font-weight:800;color:{tcolor};margin:6px 0 4px;">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Offseason roster changes driving the projection ──────────────────────────
_empty = {"adds": [], "losses": [], "rookies": []}
ch = changes.get(sel, {"off": _empty, "def": _empty})
off_ch, def_ch = ch.get("off", _empty), ch.get("def", _empty)

o_delta = trow["off_change"]           # + = more points scored (good)
d_delta = trow["def_change"]           # + = more points allowed (bad)
o_arrow = "▲" if o_delta >= 0 else "▼"
o_color = "#16a34a" if o_delta >= 0 else "#dc2626"
d_arrow = "▼" if d_delta <= 0 else "▲"  # fewer points allowed = green
d_color = "#16a34a" if d_delta <= 0 else "#dc2626"
st.markdown(
    f"#### 🔄 Offseason Impact &nbsp; "
    f"<span style='color:{o_color};font-weight:700;'>{o_arrow} {o_delta:+.1f} off. PPG</span> &nbsp;·&nbsp; "
    f"<span style='color:{d_color};font-weight:700;'>{d_arrow} {d_delta:+.1f} PPG allowed</span>",
    unsafe_allow_html=True,
)

# Offense
st.markdown("**Offense** — predicted scoring shift from acquisitions & losses")
ac1, ac2, ac3 = st.columns(3)
with ac1:
    st.markdown("**➕ Additions**")
    if off_ch["adds"]:
        for a in off_ch["adds"][:5]:
            st.markdown(f"- {a['player']} ({a['pos']}) — from {a['from']} · {a['ppr']:.0f} PPR")
    else:
        st.caption("None of note.")
with ac2:
    st.markdown("**➖ Departures**")
    if off_ch["losses"]:
        for l in off_ch["losses"][:5]:
            st.markdown(f"- {l['player']} ({l['pos']}) — {l['ppr']:.0f} PPR in '25")
    else:
        st.caption("None of note.")
with ac3:
    st.markdown("**🌟 Rookie / New Starters**")
    if off_ch["rookies"]:
        for r in off_ch["rookies"][:5]:
            st.markdown(f"- {r['player']} ({r['pos']})")
    else:
        st.caption("None projected.")

# Defense
st.markdown("**Defense** — points-allowed shift from front-7 & secondary turnover")
dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.markdown("**➕ Additions**")
    if def_ch["adds"]:
        for a in def_ch["adds"][:5]:
            st.markdown(f"- {a['player']} ({a['pos']}) — from {a['from']}")
    else:
        st.caption("None of note.")
with dc2:
    st.markdown("**➖ Departures**")
    if def_ch["losses"]:
        for l in def_ch["losses"][:5]:
            st.markdown(f"- {l['player']} ({l['pos']})")
    else:
        st.caption("None of note.")
with dc3:
    st.markdown("**🌟 New Starters**")
    if def_ch["rookies"]:
        for r in def_ch["rookies"][:5]:
            st.markdown(f"- {r['player']} ({r['pos']})")
    else:
        st.caption("None projected.")

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1, 1])

# ── Win-total distribution ────────────────────────────────────────────────────
with left:
    st.markdown("#### Win Total Distribution")
    dist = np.asarray(trow["win_dist"])
    fig = go.Figure(go.Bar(
        x=list(range(len(dist))), y=dist * 100,
        marker_color=tcolor,
        hovertemplate="%{x} wins: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                      xaxis_title="Wins", yaxis_title="Probability (%)")
    fig.add_vline(x=float(trow["proj_wins"]), line_dash="dot",
                  line_color="rgba(79,70,229,0.5)")
    st.plotly_chart(fig, use_container_width=True)

# ── Game-by-game schedule with win probability ───────────────────────────────
with right:
    st.markdown("#### 2026 Schedule & Win Odds")
    g = games[(games["home_team"] == sel) | (games["away_team"] == sel)].copy()
    g["is_home"] = g["home_team"] == sel
    g["opp"] = np.where(g["is_home"], g["away_team"], g["home_team"])
    g["win_prob"] = np.where(g["is_home"], g["p_home"], 1.0 - g["p_home"])
    g = g.sort_values("week")
    sched = pd.DataFrame({
        "Wk": g["week"].astype(int),
        "": ["vs" if h else "@" for h in g["is_home"]],
        "Opp Logo": [get_logo(o, teams_df) for o in g["opp"]],
        "Opponent": g["opp"].values,
        "Win %": (g["win_prob"] * 100).round(0).values,
    })
    st.dataframe(
        sched, hide_index=True, use_container_width=True, height=460,
        column_config={
            "Opp Logo": st.column_config.ImageColumn("", width="small"),
            "Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100,
                        format="%d%%"),
        },
    )

# ══════════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ How these projections are built"):
    if _market.get("used"):
        _power_para = (
            f"A team's **power rating** starts as its predicted net PPG centered "
            f"on the league, then is **blended {_market.get('weight', 0.0):.0%} "
            f"toward the market's implied rating** — the Vegas win total is the "
            f"most accurate public preseason signal, so the sim leans on it while "
            f"the roster model supplies the offseason-impact story above."
        )
    else:
        _power_para = "A team's **power rating** is its predicted net PPG centered on the league."
    st.markdown(f"""
**Team quality is anchored on what actually happened, not on fantasy points.**
A walk-forward backtest (2016-2025) showed that a team's prior-season point
differential is the most accurate preseason signal available to us — swapping in
opponent-adjusted ratings or EPA barely moved the error — so the model builds
each side from **2025 results, regressed toward the league mean**, and then lets
the roster do the rest.

**Offense** starts from 2025 points scored (regressed) and is **adjusted for
offseason roster turnover** — the "Off Δ" column. Each projected starter, read
straight from the **updated post-offseason depth chart** (so acquisitions and
losses are baked in), carries a regressed 2025 per-game PPR value; the change in
that roster index maps to a scoring shift that's centered league-wide so trades
redistribute rather than inflate scoring.

**Defense** works the same way: it **anchors on the team's actual 2025 points
allowed** (regressed toward the mean) and **adjusts for defensive roster
turnover** — the "Def Δ" column, built from a composite box score (sacks, TFL,
QB hits, INTs, pass breakups, forced fumbles) and centered so moves redistribute
(negative = the defense improved).

{_power_para} For every 2026 game, win probability comes from the rating gap
plus home-field advantage through a normal single-game margin model, and the
272-game slate is **simulated 20,000 times** for win totals, division-title odds,
playoff odds and expected finish (4 division winners + 3 wild cards per
conference, ties random).
    """)
