import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.styles import NFL_CSS
from utils.nav import render_sidebar_nav
from utils.espn_league import (
    espn_configured, load_league, league_name, draft_completed,
    standings_df, roster_df, matchups_df, team_names,
    load_league_season, previous_seasons, final_standings_df, draft_picks_df,
    all_time_df, h2h_df,
)
from utils.draft_grades import team_draft_grades, accuracy, graded_picks

st.set_page_config(page_title="Fantasy League · NFL", page_icon="🏆", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

render_sidebar_nav(current_page="10_Fantasy_Football_League")

if st.button("← Back to Fantasy Football", key="fl_back_btn"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🏆</div>
    <div>
        <div class="title">Fantasy Football League</div>
        <div class="subtitle">Live standings, rosters &amp; matchups from your ESPN league</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

_IMG_COL = {"Logo": st.column_config.ImageColumn("", width="small")}

# ── Not configured yet: walk the user through it ─────────────────────────────
if not espn_configured():
    st.info("This page isn't connected to an ESPN league yet.")
    st.markdown("""
**Running locally?** Add these to `.streamlit/secrets.toml`:

```toml
[espn]
league_id = 1234567890
season = 2026
espn_s2 = "..."     # only needed for private leagues
swid = "{...}"      # only needed for private leagues
```

**Deployed on Streamlit Community Cloud?** `.streamlit/secrets.toml` is
gitignored and never ships with the repo, so the same block has to be pasted
into the dashboard instead: **share.streamlit.io → your app → ⋮ → Settings →
Secrets**. The app reboots on save.

- **League ID**: the number in your league's URL
  (`fantasy.espn.com/football/team?leagueId=1234567890`).
- **espn_s2 / SWID**: cookies from a logged-in browser session at
  fantasy.espn.com — DevTools → Application/Storage → Cookies →
  `fantasy.espn.com`. Private leagues only, and they expire every few
  months — refresh them when this page starts 401-ing.
""")
    st.stop()

with st.spinner("Loading league data from ESPN..."):
    data = load_league()

if not data:
    st.stop()  # load_league() already surfaced an st.error

name = league_name(data) or "Your League"
teams_ct = len(data.get("teams", []))
season = st.secrets["espn"].get("season", "")
past = previous_seasons(data)[::-1]  # newest first

st.markdown(f"#### {name} · {season} · {teams_ct} teams")

ALL_TIME = "🏛️ All-time"
view = st.selectbox(
    "Season", [f"{season} (current)", *past, ALL_TIME] if past else [f"{season} (current)"],
    key="fl_season",
    help="Past seasons are pulled from ESPN's league history and include the "
         "full draft, so they can be graded after the fact.",
)

# ── All-time franchise history ────────────────────────────────────────────────
if view == ALL_TIME:
    seasons = tuple(sorted(past))
    st.caption(f"Across {len(seasons)} completed seasons ({min(seasons)}–{max(seasons)}). "
               "Aggregated by owner, since team names get renamed most years.")
    t_hist, t_h2h = st.tabs(["🏆 Franchise History", "⚔️ Head-to-Head"])

    with t_hist:
        at = all_time_df(seasons)
        if at.empty:
            st.info("No completed seasons available.")
        else:
            st.dataframe(
                at[["Owner", "Seasons", "W", "L", "T", "Win%", "PF", "PA",
                    "Titles", "Best", "Worst", "AvgFinish"]],
                hide_index=True, use_container_width=True,
            )
            st.caption("Best / Worst are final finishes. AvgFinish is the mean "
                       "final placing — lower is better.")

    with t_h2h:
        h = h2h_df(seasons)
        if h.empty:
            st.info("No head-to-head history available.")
        else:
            owners = sorted(h["Owner"].unique())
            who = st.selectbox("Owner", owners, key="fl_h2h_owner")
            mine = h[h["Owner"] == who].sort_values("Win%", ascending=False)
            st.dataframe(mine[["Opponent", "W", "L", "Win%"]],
                         hide_index=True, use_container_width=True)
            tot_w, tot_l = int(mine["W"].sum()), int(mine["L"].sum())
            st.caption(f"**{who}** is {tot_w}-{tot_l} all-time "
                       f"({tot_w / max(tot_w + tot_l, 1):.1%}) across every matchup.")
    st.stop()

# ── A completed past season ───────────────────────────────────────────────────
if view != f"{season} (current)":
    yr = int(view)
    hist = load_league_season(yr)
    if not hist:
        st.stop()

    t_final, t_draft, t_grade, t_sched = st.tabs(
        ["📊 Final Standings", "📝 Draft Recap", "🎯 Draft Grades", "📅 Schedule"])

    with t_final:
        fs = final_standings_df(hist)
        if fs.empty:
            st.info("No standings for this season.")
        else:
            champs = fs[fs["Finish"] == fs["Finish"].min()]
            label = "co-champions" if len(champs) > 1 else "champion"
            st.success(f"🏆 **{yr} {label}:** " + "  ·  ".join(
                f"**{c['Team']}** ({c['Owner']}) — {c['W']}-{c['L']}, {c['PF']:g} PF"
                for _, c in champs.iterrows()))
            st.dataframe(fs[["Finish", "Seed", "Logo", "Team", "Owner",
                             "W", "L", "T", "PF", "PA"]],
                         hide_index=True, use_container_width=True, column_config=_IMG_COL)
            st.caption("**Seed** is where a team finished the regular season; "
                       "**Finish** is where it ended up after the playoffs.")

    with t_draft:
        dp = draft_picks_df(hist, yr)
        if dp.empty:
            st.info("No draft data for this season.")
        else:
            show = dp[["Overall", "Rd", "Pick", "Team", "player", "pos",
                       "nfl_team", "espn_proj_pts", "actual_pts"]].copy()
            show.columns = ["Ovr", "Rd", "Pk", "Team", "Player", "Pos",
                            "NFL", "ESPN Proj", "Actual"]
            st.dataframe(show, hide_index=True, use_container_width=True, height=520)
            st.caption("**ESPN Proj** is ESPN's preseason projection; **Actual** is "
                       "what the player really scored that season.")

    with t_grade:
        st.markdown("##### How the draft looked to each of us — and what happened")
        g = team_draft_grades(yr)
        if g.empty:
            st.info("Not enough data to grade this draft.")
        else:
            espn_ok = not g["espn_pts"].isna().all()
            cols = ["Team", "My Rank"] + (["ESPN Rank"] if espn_ok else []) + \
                   ["Actual Rank", "Finish", "model_pts"] + \
                   (["espn_pts"] if espn_ok else []) + ["actual_pts", "graded", "Unrated"]
            disp = g[cols].rename(columns={
                "model_pts": "My Proj", "espn_pts": "ESPN Proj",
                "actual_pts": "Actual Pts", "graded": "Graded"})
            st.dataframe(disp, hide_index=True, use_container_width=True)

            if not espn_ok:
                st.warning(f"ESPN no longer publishes preseason projections for {yr}, "
                           "so only this model is graded here.")

            acc = accuracy(yr)
            if acc:
                st.markdown("**Did the draft actually predict the season?**")
                mcols = st.columns(2 if espn_ok else 1)
                for col, (label, key) in zip(mcols, [("My model", "model"), ("ESPN", "espn")]):
                    if key not in acc:
                        continue
                    col.metric(f"{label} → points scored", f"{acc[key]['vs_points']:+.2f}")
                    col.metric(f"{label} → final standings", f"{acc[key]['vs_finish']:+.2f}")
                st.caption(
                    "Spearman correlation, +1.00 = perfect, 0.00 = no signal. Both "
                    "sides are graded on the same players, using only preseason "
                    "information — this model never sees the season it is projecting."
                )
            gp = graded_picks(yr)
            n_un = int((~gp["comparable"]).sum())
            if n_un:
                st.caption(
                    f"{n_un} picks are excluded: the model projects each season from "
                    "the previous one, so rookies have no projection. Grading them "
                    "would penalise whoever drafted rookies, so they're dropped from "
                    "both sides. Kickers and defenses are excluded for the same reason."
                )

    with t_sched:
        mdf = matchups_df(hist)
        if mdf.empty:
            st.info("No schedule for this season.")
        else:
            weeks = sorted(mdf["week"].dropna().unique())
            wk_sel = st.selectbox("Week", weeks, key="fl_hist_week")
            for _, m in mdf[mdf["week"] == wk_sel].iterrows():
                c1, c2, c3 = st.columns([3, 1, 3])
                won_h = m["winner"] == "HOME"
                c1.markdown(f"{'**' if won_h else ''}{m['home_team']}{'**' if won_h else ''}")
                c1.caption(f"{m['home_pts']:g} pts")
                c2.markdown("<div style='text-align:center'>vs</div>", unsafe_allow_html=True)
                if m["away_team"] == "BYE":
                    c3.markdown("*BYE*")
                else:
                    won_a = m["winner"] == "AWAY"
                    c3.markdown(f"{'**' if won_a else ''}{m['away_team']}{'**' if won_a else ''}")
                    c3.caption(f"{m['away_pts']:g} pts")
                st.divider()
    st.stop()

# ── Current season ────────────────────────────────────────────────────────────
if not draft_completed(data):
    st.warning("The draft for this league hasn't happened yet — standings will "
                "show all-zero records and rosters will be empty until it does. "
                "This page will fill in automatically once the season starts.")

tab_standings, tab_rosters, tab_schedule = st.tabs(["📊 Standings", "🧢 Rosters", "📅 Schedule"])

# ── Standings ─────────────────────────────────────────────────────────────────
with tab_standings:
    sdf = standings_df(data)
    if sdf.empty:
        st.info("No standings available.")
    else:
        show = sdf[["Rank", "Logo", "Team", "Owner", "W", "L", "T", "PF", "PA", "Streak"]]
        st.dataframe(show, hide_index=True, use_container_width=True, column_config=_IMG_COL)

# ── Rosters ───────────────────────────────────────────────────────────────────
with tab_rosters:
    names = team_names(data)
    team_ids = sorted(names.keys())

    # Default to the team owned by the logged-in SWID, if we can find it.
    my_swid = st.secrets["espn"].get("swid", "")
    my_team_id = next(
        (t["id"] for t in data.get("teams", []) if my_swid in t.get("owners", [])),
        team_ids[0] if team_ids else None,
    )

    rhdr, rsel = st.columns([2, 1])
    with rsel:
        sel_team = st.selectbox(
            "View roster", team_ids,
            index=team_ids.index(my_team_id) if my_team_id in team_ids else 0,
            format_func=lambda tid: f"You — {names[tid]}" if tid == my_team_id else names[tid],
            key="fl_roster_team", label_visibility="collapsed",
        )
    with rhdr:
        st.markdown(f"#### {'Your Roster — ' if sel_team == my_team_id else ''}{names.get(sel_team, '')}")

    rdf = roster_df(data, sel_team)
    if rdf.empty:
        st.info("No players rostered yet (draft hasn't happened).")
    else:
        st.dataframe(rdf, hide_index=True, use_container_width=True)

# ── Schedule / matchups ────────────────────────────────────────────────────────
with tab_schedule:
    mdf = matchups_df(data)
    if mdf.empty:
        st.info("No schedule available.")
    else:
        weeks = sorted(mdf["week"].dropna().unique())
        cur_week = data.get("scoringPeriodId", weeks[0] if weeks else 1)
        sel_week = st.selectbox(
            "Week", weeks,
            index=weeks.index(cur_week) if cur_week in weeks else 0,
            key="fl_sched_week",
        )
        wk = mdf[mdf["week"] == sel_week]
        for _, m in wk.iterrows():
            c1, c2, c3 = st.columns([3, 1, 3])
            c1.markdown(f"**{m['home_team']}**")
            c1.caption(f"{m['home_pts']:g} pts")
            c2.markdown("<div style='text-align:center'>vs</div>", unsafe_allow_html=True)
            if m["away_team"] == "BYE":
                c3.markdown("*BYE*")
            else:
                c3.markdown(f"**{m['away_team']}**")
                c3.caption(f"{m['away_pts']:g} pts")
            st.divider()
