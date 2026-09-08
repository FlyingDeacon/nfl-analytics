import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from utils.styles import NFL_CSS
from utils.data_loader import (load_teams, load_schedules, get_logo, get_base_dir,
                               _file_mtime)
from utils.chopped_log import (COLUMNS, clear_pick, load_picks, picks_path,
                               record_pick, save_picks)
from utils.survivor import (compassion_default, current_week, grade_picks,
                            optimal_plan, survival_probability, week_deadline,
                            week_options)
from utils.projection import matchup_tables, input_paths
from utils.nav import render_sidebar_nav, render_last_updated
from utils.gate import require_passcode

st.set_page_config(page_title="CHOPPED Survivor · NFL", page_icon="🔪", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

require_passcode("CHOPPED Survivor")

render_sidebar_nav(current_page="13_Chopped_Survivor")

# Two entries into one pot are not two tickets unless they can fail separately,
# so each side is given a different job rather than the same optimiser twice.
# Edit the display names here; the ids are what the pick log is keyed on and
# changing one would orphan that entry's history.
ENTRIES = (("brandon", "Brandon", "**Chalk.** Take the mathematically best pick every week."),
           ("wife", "Wife", "**Hedge.** Take the best team the other entry is not using."))

LAST_WEEK = 18
RESULT_ICON = {"win": "✅", "loss": "❌", "pending": "⏳"}
# The Goofball Compassion Clause only covers a missed pick through this week.
COMPASSION_THROUGH = 5

if st.button("← Back to Season Projections", key="cs_back"):
    st.switch_page("pages/9_Record_Predictions.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔪</div>
    <div>
        <div class="title">CHOPPED Survivor</div>
        <div class="subtitle">One pick a week · no team twice · a loss ends your season · last entrant takes the pot</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)
render_last_updated(*input_paths())

_base = get_base_dir()
_SCHED_KEY = _file_mtime(_base / "data/raw/schedules.csv")
teams_df = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))
sched = load_schedules(_mtime=_SCHED_KEY)
_, tw = matchup_tables()

# Cache key for everything derived from the projection. It has to be a real
# hashed argument on the cached functions below: Streamlit does NOT hash
# parameters whose names start with an underscore, so a cache built on
# _week/_used style arguments returns its very first result forever — which is
# what used to make this page show Week 1's board no matter which week you set.
PROJ_KEY = tuple(_file_mtime(p) for p in input_paths())

ALL_TEAMS = sorted(tw["team"].unique())


def _crest(abbr: str, size: int = 26) -> str:
    url = get_logo(abbr, teams_df)
    return (f'<img src="{url}" width="{size}" style="vertical-align:middle;">'
            if url else f"<b>{abbr}</b>")


@st.cache_data(show_spinner="Pricing every legal pick…")
def _options(used: frozenset, week: int, pool: int, key: tuple):
    return week_options(tw, set(used), week, pool_size=pool)


@st.cache_data(show_spinner=False)
def _plan(used: frozenset, week: int, key: tuple):
    return optimal_plan(tw, set(used), week)


@st.cache_data(show_spinner=False)
def _plan_around(used: frozenset, week: int, team: str, key: tuple):
    """The rest of the season re-solved around a pick we have committed to.

    Needed because the recommendation and the plan below it have to describe the
    same season: once an entry is steered off the unconstrained optimum, the
    weeks after it change too, and showing the old plan would quietly contradict
    the pick sitting above it.
    """
    return optimal_plan(tw, set(used), week, forced=(week, team))


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS  (sidebar, so the weekly decision owns the top of the page)
# ══════════════════════════════════════════════════════════════════════════════
LIVE_WEEK = current_week(sched)

st.sidebar.markdown('<div class="sidebar-filters-label">CHOPPED</div>',
                    unsafe_allow_html=True)
cur_week = st.sidebar.number_input(
    "Week", min_value=1, max_value=LAST_WEEK, value=LIVE_WEEK, step=1,
    key="cs_week",
    help=f"Defaults to the live week ({LIVE_WEEK}) off the schedule. Change it to "
         f"look ahead — nothing you do here writes a pick.")
cur_week = int(cur_week)
pool_size = int(st.sidebar.number_input(
    "Entries in the pool", min_value=2, max_value=1000, value=50, step=1,
    key="cs_pool",
    help="Drives the EV column. Pot share is what you are actually playing for, "
         "and how much being different is worth depends entirely on how many "
         "people you would be splitting with."))
diverge = st.sidebar.toggle(
    "Split the two entries", value=True, key="cs_diverge",
    help="Both entries solve the same schedule, so left alone they converge on the "
         "same answer — and get knocked out by the same upset. This steers the "
         "second entry to its best pick that the first is not using.")

# ══════════════════════════════════════════════════════════════════════════════
# SEASON STATE
# ══════════════════════════════════════════════════════════════════════════════
picks = load_picks()
graded = grade_picks(picks, sched)

state = {}
for eid, name, _ in ENTRIES:
    mine = graded[graded["entry"] == eid].sort_values("week")
    lost = mine[mine["result"] == "loss"]
    this_week = mine[mine["week"] == cur_week]
    state[eid] = {
        "name": name,
        "history": mine,
        "used": set(mine["team"]),
        # Teams unavailable when solving THIS week: everything already spent in
        # some other week. The current week's own pick is not a constraint on it.
        "spent_elsewhere": frozenset(mine[mine["week"] != cur_week]["team"]),
        "out_week": int(lost["week"].min()) if not lost.empty else None,
        "locked": this_week.iloc[0] if not this_week.empty else None,
    }

alive = [s for s in state.values()
         if s["out_week"] is None or s["out_week"] >= cur_week]

# ── This week at a glance ─────────────────────────────────────────────────────
deadline = week_deadline(sched, cur_week)
_when = (f"{deadline:%a %b} {deadline.day}, "
         f"{(deadline.hour % 12) or 12}:{deadline.minute:02d} "
         f"{'AM' if deadline.hour < 12 else 'PM'} ET"
         if deadline is not None else "schedule not loaded")

if cur_week <= COMPASSION_THROUGH:
    _default = compassion_default(sched, cur_week)
    _miss_value = _default or "—"
    _miss_sub = (f"The Compassion Clause defaults you to {_default}, the home team of "
                 f"this week's last game. One time only, and it expires after "
                 f"Week {COMPASSION_THROUGH}.")
else:
    _miss_value = "You're out"
    _miss_sub = (f"Past Week {COMPASSION_THROUGH} a missed pick is an automatic loss and "
                 "the commissioner removes your best remaining team.")

b1, b2, b3 = st.columns(3)
b1.markdown(
    f'<div class="stat-card"><div class="label">Picking</div>'
    f'<div class="value">Week {cur_week} of {LAST_WEEK}</div>'
    f'<div class="sub">Locks {_when}</div></div>', unsafe_allow_html=True)
b2.markdown(
    f'<div class="stat-card"><div class="label">Still alive</div>'
    f'<div class="value">{len(alive)} of {len(ENTRIES)}</div>'
    f'<div class="sub">{", ".join(s["name"] for s in alive) or "both entries are out"}'
    f'</div></div>', unsafe_allow_html=True)
b3.markdown(
    f'<div class="stat-card"><div class="label">If you forget to pick</div>'
    f'<div class="value">{_miss_value}</div>'
    f'<div class="sub">{_miss_sub}</div></div>', unsafe_allow_html=True)

if cur_week != LIVE_WEEK:
    st.caption(f"👀 Looking ahead — the live week is **{LIVE_WEEK}**. "
               "Picks can only be locked for the live week.")

st.info(
    "**The rule that decides everything:** a team can only be used once all season. "
    "So the best pick this week is rarely the biggest favourite — it is the one that "
    "leaves the strongest set of teams for the weeks still to come. Every option below "
    "is priced that way: **Cost** is how much full-season survival you give up versus "
    "the optimal choice, so a 0.00 cost is the pick that keeps the most value in reserve.",
    icon="🧠",
)

# ══════════════════════════════════════════════════════════════════════════════
# THE TWO ENTRIES
# ══════════════════════════════════════════════════════════════════════════════
recs = {}
taken = None      # entry 1's pick this week, so entry 2 can be steered off it

for col, (eid, name, style) in zip(st.columns(2), ENTRIES):
    s = state[eid]
    with col:
        st.markdown(f"### {name}")
        st.caption(style)

        # ── Out of the pool ───────────────────────────────────────────────────
        if s["out_week"] is not None and s["out_week"] < cur_week:
            gone = s["history"][s["history"]["week"] == s["out_week"]].iloc[0]
            st.error(f"Knocked out in Week {s['out_week']} — **{gone['team']}** lost to "
                     f"{gone['opponent']}.", icon="💀")
            st.caption(f"{len(s['used'])} teams spent. Nothing left to decide, but the "
                       "log below still shows how the run went.")
            continue

        st.caption(f"{len(ALL_TEAMS) - len(s['used'])} teams still unspent · "
                   f"{len(s['used'])} used")

        # ── Already locked for this week ──────────────────────────────────────
        if s["locked"] is not None:
            lk = s["locked"]
            icon = RESULT_ICON.get(lk["result"], "⏳")
            score = (f'{int(lk["points"])}–{int(lk["opp_points"])}'
                     if lk["result"] != "pending" else
                     f'{"vs" if lk["is_home"] else "@"} {lk["opponent"]}')
            st.markdown(
                f'<div class="stat-card" style="padding:16px 12px;">'
                f'<div class="label">Locked · Week {cur_week}</div>'
                f'<div style="font-size:1.8rem;font-weight:800;margin:6px 0 2px;">'
                f'{_crest(lk["team"], 34)} {lk["team"]} {icon}</div>'
                f'<div class="sub">{score}</div></div>', unsafe_allow_html=True)
            taken = taken or lk["team"]
            recs[name] = lk["team"]

            if lk["result"] == "pending":
                if st.button("Unlock this pick", key=f"cs_unlock_{eid}"):
                    clear_pick(cur_week, eid)
                    st.rerun()

            # No forward plan for a pick that has already lost — this entry's
            # season ended on the card above, not in Week cur_week + 1.
            rest = (pd.DataFrame() if lk["result"] == "loss"
                    else _plan(frozenset(s["used"]), cur_week + 1, PROJ_KEY))
            if not rest.empty:
                with st.expander(f"Plan from Week {cur_week + 1} "
                                 f"({survival_probability(rest):.1%} to run the table)"):
                    st.dataframe(
                        rest.assign(**{
                            "Matchup": rest.apply(
                                lambda r: f'{r["team"]} {"vs" if r["is_home"] else "@"} '
                                          f'{r["opponent"]}', axis=1),
                            "Win %": (rest["win_prob"] * 100).round(0).astype(int),
                        })[["week", "Matchup", "Win %"]].rename(columns={"week": "Wk"}),
                        hide_index=True, use_container_width=True)
            continue

        # ── Still to pick ─────────────────────────────────────────────────────
        opts = _options(s["spent_elsewhere"], cur_week, pool_size, PROJ_KEY)
        if diverge and taken is not None:
            opts = opts[opts["team"] != taken].reset_index(drop=True)
        if opts.empty:
            st.warning("No legal picks left for this week.")
            continue

        best = opts.iloc[0]
        recs[name] = best["team"]
        if taken is None:
            taken = best["team"]

        _toll = (f'<div class="sub" style="margin-top:4px;">Costs '
                 f'{best["cost_vs_best"]:.2%} of season survival to stay off {taken}</div>'
                 if diverge and taken != best["team"] and best["cost_vs_best"] > 0 else "")
        st.markdown(
            f'<div class="stat-card" style="padding:16px 12px;">'
            f'<div class="label">Recommended · Week {cur_week}</div>'
            f'<div style="font-size:1.8rem;font-weight:800;margin:6px 0 2px;">'
            f'{_crest(best["team"], 34)} {best["team"]}</div>'
            f'<div class="sub">{best["win_prob"]:.0%} to win '
            f'{"vs" if best["is_home"] else "@"} {best["opponent"]}</div>{_toll}</div>',
            unsafe_allow_html=True)

        # The recommendation is a recommendation, not a lock — the selectbox
        # defaults to it but lets a gut call be recorded, because a pick made in
        # the app and not written down is the one that gets forgotten by Sunday.
        labels = {r["team"]: (f'{r["team"]} · {r["win_prob"]:.0%} '
                              f'{"vs" if r["is_home"] else "@"} {r["opponent"]} '
                              f'(cost {r["cost_vs_best"] * 100:.2f})')
                  for _, r in opts.iterrows()}
        choice = st.selectbox("Pick to lock in", list(labels), index=0,
                              format_func=labels.get, key=f"cs_choice_{eid}")
        if cur_week == LIVE_WEEK:
            if st.button(f"🔒 Lock in {choice} for Week {cur_week}",
                         key=f"cs_lock_{eid}", type="primary",
                         use_container_width=True):
                record_pick(cur_week, eid, choice)
                st.rerun()
        else:
            st.button(f"🔒 Lock in {choice}", key=f"cs_lock_{eid}",
                      disabled=True, use_container_width=True,
                      help=f"Set the sidebar week back to {LIVE_WEEK} to lock a pick.")

        show = opts.head(8).copy()
        show["Matchup"] = show.apply(
            lambda r: f'{r["team"]} {"vs" if r["is_home"] else "@"} {r["opponent"]}', axis=1)
        show["Win %"] = (show["win_prob"] * 100).round(0).astype(int)
        show["Cost"] = (show["cost_vs_best"] * 100).round(2)
        show["Field %"] = (show["popularity"] * 100).round(0).astype(int)
        show["EV"] = show["pot_ev"].round(2)
        st.dataframe(show[["Matchup", "Win %", "Cost", "Field %", "EV"]], hide_index=True,
                     use_container_width=True,
                     column_config={
                         "Win %": st.column_config.NumberColumn(
                             "Win %", format="%d%%", help="Chance this team wins this week"),
                         "Cost": st.column_config.NumberColumn(
                             "Cost", format="%.2f",
                             help="Season survival given up versus the optimal pick, "
                                  "in percentage points. 0.00 is the optimum."),
                         "Field %": st.column_config.NumberColumn(
                             "Field %", format="%d%%",
                             help="Estimated share of the pool on this team. Modelled "
                                  "from the win probability, not read off a real grid."),
                         "EV": st.column_config.NumberColumn(
                             "EV", format="%.2f",
                             help="Expected share of the pot, where 1.00 is the pool "
                                  "average. Above 1.00 means you gain ground on the "
                                  "field in the weeks you survive."),
                     })

        # Survival and pot share can disagree, and when they do it is worth
        # saying out loud rather than burying in a column — but it is a judgement
        # call about how much variance you want, so it is surfaced, not obeyed.
        # Only when the edge is real: the EV leader is often ahead by less than a
        # rounding step, and trading eight points of win probability for 0.6% of
        # the pot is not a decision worth putting in front of anyone.
        EV_EDGE_MIN = 0.02
        ev_best = opts.loc[opts["pot_ev"].idxmax()]
        if (ev_best["team"] != best["team"]
                and ev_best["pot_ev"] - best["pot_ev"] >= EV_EDGE_MIN):
            st.caption(
                f'⚖️ Highest **EV** this week is **{ev_best["team"]}** '
                f'({ev_best["pot_ev"]:.2f} vs {best["pot_ev"]:.2f}) — only '
                f'{ev_best["win_prob"]:.0%} to win, but just '
                f'{ev_best["popularity"]:.0%} of the field is on it, so it gains '
                f'the most ground when it hits. The pick above is still the '
                f'survival-maximising one.')

        plan_df = _plan_around(s["spent_elsewhere"], cur_week, best["team"], PROJ_KEY)
        if not plan_df.empty:
            with st.expander(f"Full plan to Week {LAST_WEEK} "
                             f"({survival_probability(plan_df):.1%} to run the table)"):
                p = plan_df.copy()
                p["Matchup"] = p.apply(
                    lambda r: f'{r["team"]} {"vs" if r["is_home"] else "@"} {r["opponent"]}',
                    axis=1)
                p["Win %"] = (p["win_prob"] * 100).round(0).astype(int)
                st.dataframe(p[["week", "Matchup", "Win %"]].rename(columns={"week": "Wk"}),
                             hide_index=True, use_container_width=True)
                st.caption(
                    "This entry's own best path from here, so the two plans will show the "
                    "same team in some later week. That is fine — divergence is a weekly "
                    "decision, not a fixed schedule. Lock this week's picks and the split "
                    "is recomputed next week against what you actually spent.")

if len(recs) == 2 and len(set(recs.values())) == 1:
    st.warning(
        f"Both entries are on **{list(recs.values())[0]}**. Only one of you can win the pot, "
        "so playing the same team means one upset eliminates you both. Turn on "
        "*Split the two entries* in the sidebar to separate them."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PICK LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📜 Pick log")

if graded.empty:
    st.caption("Nothing locked in yet. Picks land here as soon as you lock them, and "
               "the result fills in on its own once the scores reach the schedule file.")
else:
    log = graded.copy()
    log["cell"] = log.apply(
        lambda r: f'{RESULT_ICON.get(r["result"], "")} {r["team"]}'
                  + (f' {int(r["points"])}–{int(r["opp_points"])}'
                     if r["result"] != "pending" else
                     f' {"vs" if r["is_home"] else "@"} {r["opponent"]}'),
        axis=1)
    grid = (log.pivot_table(index="week", columns="entry", values="cell",
                            aggfunc="first")
               .reindex(columns=[e[0] for e in ENTRIES])
               .rename(columns={e[0]: e[1] for e in ENTRIES})
               .fillna("—")
               .reset_index()
               .rename(columns={"week": "Wk"}))
    st.dataframe(grid, hide_index=True, use_container_width=True)
    st.caption("Results are read off the schedule, not typed in — a tie counts as a win, "
               "per the league rule. Run `scripts/update_2026_data.py` to pull the "
               "weekend's scores, then hit Refresh Data in the sidebar.")

with st.expander("💾 Backup, restore and hand-editing"):
    st.markdown(
        f"The log lives at `{picks_path().relative_to(get_base_dir())}`. On the deployed "
        "app that file survives a browser refresh but **not** a redeploy or an idle "
        "restart, so download it after locking picks and commit it to the repo when you "
        "want a pick to be permanent.")
    st.download_button(
        "⬇️ Download the pick log",
        data=picks.to_csv(index=False).encode(),
        file_name="chopped_picks.csv", mime="text/csv", key="cs_dl")
    up = st.file_uploader("Restore from a downloaded copy", type="csv", key="cs_up")
    if up is not None:
        try:
            incoming = pd.read_csv(up)
        except Exception as err:
            st.error(f"Could not read that file: {err}")
        else:
            if not set(COLUMNS) <= set(incoming.columns):
                st.error(f"That CSV needs the columns {', '.join(COLUMNS)}.")
            else:
                st.dataframe(incoming[COLUMNS], hide_index=True,
                             use_container_width=True)
                # Behind a button, not automatic: st.file_uploader keeps handing
                # the same file back on every rerun, which would silently undo
                # any pick locked after the upload.
                if st.button("Replace the log with this file", key="cs_restore"):
                    save_picks(incoming)
                    st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SEASON MAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🗺️ Season map")
st.caption(f"The schedule from Week {cur_week} on, recomputed from the current "
           "projection so it stays true as the season moves.")

_rest = tw[tw["week"] >= cur_week]

# Weeks where even the best available team is shaky are the ones that decide the
# pool — you have to arrive at them still holding somebody good.
_danger = (_rest.groupby("week")["win_prob"].max()
           .sort_values().head(3).mul(100).round(0).astype(int))
# Teams that are heavy favourites often are the resource being rationed; teams
# that are heavy dogs often are the opponents worth attacking.
_premium = (_rest[_rest["win_prob"] >= 0.70].groupby("team").size()
            .sort_values(ascending=False).head(8))
_targets = (_rest[_rest["win_prob"] <= 0.30].groupby("team").size()
            .sort_values(ascending=False).head(6))


def _spent_by(team: str) -> str:
    """Which entries have already burned this team."""
    who = [s["name"] for s in state.values() if team in s["used"]]
    return ", ".join(who) if who else "—"


m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**Danger weeks**")
    st.caption("Best pick available is weakest here")
    if _danger.empty:
        st.caption("No weeks left.")
    else:
        st.dataframe(pd.DataFrame({"Week": _danger.index,
                                   "Best available": [f"{v}%" for v in _danger.values]}),
                     hide_index=True, use_container_width=True)
with m2:
    st.markdown("**Premium teams**")
    st.caption("Weeks left where they are 70%+ favourites")
    if _premium.empty:
        st.caption("Nobody is a 70%+ favourite in the weeks that remain.")
    else:
        st.dataframe(pd.DataFrame({"Team": _premium.index, "Good weeks": _premium.values,
                                   "Spent by": [_spent_by(t) for t in _premium.index]}),
                     hide_index=True, use_container_width=True)
with m3:
    st.markdown("**Teams to attack**")
    st.caption("Weeks left where they are 30%-or-worse dogs")
    if _targets.empty:
        st.caption("No heavy underdogs left on the board.")
    else:
        st.dataframe(pd.DataFrame({"Team": _targets.index, "Bad weeks": _targets.values}),
                     hide_index=True, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 How to play it")

_worst_week = int(_danger.index[0]) if not _danger.empty else cur_week
_worst_pct = f"{_danger.iloc[0]}%" if not _danger.empty else "n/a"
_hoarded = _premium.index[0] if not _premium.empty else "the best team on the board"
_top_premium = (", ".join(f"{t} ({n})" for t, n in _premium.head(4).items())
                or "nobody, this late")
_top_targets = (", ".join(f"{t} ({n})" for t, n in _targets.head(4).items())
                or "nobody, this late")

_a, _b = st.columns(2)
with _a:
    st.markdown(f"""
**{ENTRIES[0][1]} — play the chalk**

Take the 0.00-cost pick every single week, no exceptions. This is the highest
survival path that exists given the schedule, and its whole value is that it never
gets clever. You are not trying to be different from the pool here; you are trying
to still be alive in Week {_worst_week}, and most entrants will not be, because they
will have spent {_hoarded} on an early blowout they did not need.

The discipline this asks for is refusing an 82% week when the tool says 78%. That
gap is not the tool being wrong — it is the tool charging you for the team you would
have burned.
""")
with _b:
    st.markdown(f"""
**{ENTRIES[1][1]} — play the hedge**

Take the best team the other entry is not using. It will usually cost a fraction of a
percent this week, and it buys the only thing that matters with two entries: you
cannot both die to the same upset. If the chalk pick goes down on a last-second
field goal, one of you is still standing.

Over the season this naturally builds a different inventory of remaining teams, so
by Week 8 the two entries are not near-copies with one swap — they are genuinely
covering different outcomes.
""")

st.markdown(f"""
**Both entries, same two habits**

**1 · Spend the cheap wins, hoard the expensive ones.** Almost every good week comes
from playing somebody against the same handful of bad teams — right now that is
{_top_targets}. The premium teams ({_top_premium}) are a limited resource. Burning one
for an 81% week when you could have had 78% from a team you will never want again is
how people lose this pool.

**2 · Work backwards from Week {_worst_week}.** It is the thinnest week left: the best
team available is only {_worst_pct}. Whoever you are saving for a rainy day, that
is the day. Do not arrive there holding only teams you were avoiding.
""")

with st.expander("The rules that make those habits pay"):
    st.markdown(f"""
**There is no forgiven loss.** A pick that loses ends your season. The only safety net
is the Goofball Compassion Clause, and it covers a pick you *forgot to submit* through
Week {COMPASSION_THROUGH} — it defaults you to the home team of that week's last game,
one time only. From **Week {COMPASSION_THROUGH + 1}** on, a missed or invalid pick is an
automatic loss *and* the commissioner removes your best remaining team, which is a
double penalty: you are out, and on the way out you lose your answer to the thin weeks.

**The tiebreaker rewards the same discipline.** Ties are broken by the combined wins of
your **remaining** teams divided by how many you have left. Hoarding strong teams both
raises your late-season floor and wins ties, so there is no tension between playing to
survive and playing to win the tiebreak — it is one plan, not two.

**Ties on the field count as wins.** A pick only fails if your team actually loses,
which slightly favours taking a road favourite over passing on a week.
""")

st.caption(
    "Survival percentages look brutal because running 18 straight weeks is genuinely hard — "
    "but every entrant faces the same gauntlet, and you only need to outlast them, not the "
    "schedule."
)
