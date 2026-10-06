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
from utils.chopped_field import leaderboard, load_field
from utils.survivor import (compassion_default, current_week,
                            expected_season_wins, grade_picks, optimal_plan,
                            survival_probability, week_deadline, week_options)
from utils.projection import matchup_tables, input_paths
from utils.nav import render_sidebar_nav, render_last_updated
from utils.gate import require_passcode

st.set_page_config(page_title="CHOPPED Survivor · NFL", page_icon="🔪", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# The three summary cards carry text of wildly different lengths ("2 of 2" next to
# a two-sentence Compassion Clause note), so left alone they end at three different
# heights.
#
# They are laid out as a single CSS grid rather than as three st.columns. Going
# through Streamlit's columns means the card only fills its slot if a percentage
# height resolves through four nested Streamlit wrappers, and whether a stretched
# flex item counts as a definite height for that purpose is exactly the sort of
# thing engines disagree on — it measured dead even in Chromium while still
# rendering ragged in Safari. A grid row stretches its items by default, with no
# height chain to resolve, so the three stay equal everywhere. It also sidesteps
# the 320px min-width Streamlit puts on every column, which used to wrap the row
# 2-then-1 and leave the third card full-width and short.
st.markdown("""
<style>
.cs-band {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 6px;
}
.cs-band .stat-card {
    margin: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
@media (max-width: 640px) {
    .cs-band { grid-template-columns: 1fr; }
}
/* The two pick cards sit above a stack of controls, so they cannot be grown to
   the column height the way the summary cards are — they would swallow the page.
   They are instead given a floor tall enough for the tallest of the three states
   they come in (recommended with two reasons, locked with a score, chopped), in
   rem so it tracks the font size. */
.stat-card.cs-pick {
    min-height: 10rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 16px 12px;
    margin: 0 0 6px;
}
</style>
""", unsafe_allow_html=True)

require_passcode("CHOPPED Survivor")

render_sidebar_nav(current_page="13_Chopped_Survivor")

# Edit the display names here; the ids are what the pick log is keyed on and
# changing one would orphan that entry's history.
ENTRIES = (("blake", "Blake"), ("alaina", "Alaina"))
# Whose pick is solved first, which is not the order they are displayed in.
# First claim goes to the entry with the least margin for error — the one with
# no mulligan left — so it is never the fragile side that gets pushed off the
# team it needed.
PRIORITY = ("alaina", "blake")

# ── What the recommendation is actually maximising ───────────────────────────
# Both entries now play for the pot rather than for survival, because in this
# pool they are not the same objective and survival is the weaker one. Against
# the field's real remaining teams, 8000 simulated seasons put a pure
# survival-maximiser last among every non-greedy rule tried: it reaches Week 18
# most often and wins least often, because it gets there in a crowd and then
# loses the tiebreak.
#
# The rule is: among picks that keep at least this fraction of the best season
# still available, take the best combination of pot EV and tiebreaker.
#
# The gate is season survival, not "within 8 points of this week's safest team"
# as it used to be. A win-probability gate says nothing about what a pick costs
# the rest of the season, which is the only currency worth rationing.
#
# It is a *ratio* deliberately. Survival is 15% in Week 4 and single digits by
# Week 12, so a budget of "0.8 percentage points" silently goes from a 5% detour
# to a 30% one as the season shortens — the same constant would mean something
# different every week. A ratio holds its meaning.
#
# One gate per entry, because the two are not in the same race. Blake has his
# mulligan and reaches Week 18 about one season in nine; Alaina spent hers in
# Week 1, so her next loss ends it and she gets there about one in fifty. Asking
# a single constant to serve both costs real money: swept together they peak at
# 10.5%, swept apart at 12.2%.
#
# The direction is the opposite of the intuition. The entry with no safety net
# is *not* the one to gamble with — Alaina tightens to 0.94, because her only
# realistic path to the pot is surviving into the tiebreak room, and a lottery
# ticket that is already losing does not improve by buying worse odds. Blake,
# who can absorb one loss, is the one who can afford to chase ground.
#
# Both numbers are peaks of a sweep scored the way the pot is actually won: both
# entries in the same simulated pool against the league's real surviving field
# read off the sheet, counting a win when *either* of them takes it. Week 5,
# 12,000 seasons x 6 seeds; the winning pair's worst seed (11.95%) beats every
# other pair's best.
#
# Treat these as re-swept weekly, not as constants. The surface is rugged, and
# Blake's in particular is a knife-edge — 0.80 scores 8.8% and 0.84 scores 6.5%
# against 12.2% at 0.82. Alaina's is flatter (0.88-1.00 all land between 10.7%
# and 12.2%). What has been stable across every sweep is only the shape: gate
# hard enough to stay in the room, then win the room on the tiebreaker.
SURVIVAL_KEEP = {"blake": 0.82, "alaina": 0.94}
AGGRESSION_FLOOR = 0.60

# Price of a projected win burnt off the tiebreaker, in units of pot EV. The
# tiebreaker decides roughly half of the seasons where anyone survives at all
# (the median finish has two or three entries standing), and it reduces to the
# average final wins of the teams you never spent — so retiring a 4-win team
# instead of an 11-win one is worth real money even when both are safe picks
# this Sunday. Anything in 0.01-0.04 picks the same plan; 0 costs about two
# thirds of the simulated win rate, which is the only part of this that matters.
TIEBREAK_WEIGHT = 0.02

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
        <div class="subtitle">One pick a week · no team twice · one mulligan · winner takes the whole pot</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)
render_last_updated(*input_paths())

_base = get_base_dir()
_SCHED_KEY = _file_mtime(_base / "data/raw/schedules.csv")
teams_df = load_teams(mtime=_file_mtime(_base / "data/raw/teams.csv"))
sched = load_schedules(mtime=_SCHED_KEY)
_, tw = matchup_tables()

# Cache key for everything derived from the projection. It has to be a real
# hashed argument on the cached functions below: Streamlit does NOT hash
# parameters whose names start with an underscore, so a cache built on
# _week/_used style arguments returns its very first result forever — which is
# what used to make this page show Week 1's board no matter which week you set.
PROJ_KEY = tuple(_file_mtime(p) for p in input_paths())

ALL_TEAMS = sorted(tw["team"].unique())
# Projected final wins per team — the units the tiebreaker is settled in, and
# so a price on every pick over and above what it does for you this Sunday.
TEAM_WINS = expected_season_wins(tw, sched)


def _crest(abbr: str, size: int = 26) -> str:
    url = get_logo(abbr, teams_df)
    return (f'<img src="{url}" width="{size}" style="vertical-align:middle;">'
            if url else f"<b>{abbr}</b>")


def _as_blocked(pairs: tuple) -> dict:
    """(week, team) pairs -> {week: {teams}}, the planner's blocking format.

    Kept as a flat tuple at the call sites because st.cache_data has to hash it
    and a dict of sets is not hashable.
    """
    out = {}
    for week, team in pairs:
        out.setdefault(week, set()).add(team)
    return out


# The sheet is the league's own record and changes only when someone picks, so
# a long TTL is plenty. The key is bumped by the hour rather than being left to
# the TTL alone so that Reload From Disk does not silently serve a stale board.
#
# "America/New_York", not "US/Eastern": the two name the same zone, but the
# latter is a backward-compatibility alias that the slim tzdata on Streamlit
# Cloud's Python 3.14 image does not ship, and looking it up there takes the
# whole page down. The rest of the app already uses the canonical name.
_SHEET_TTL_KEY = pd.Timestamp.now(tz="America/New_York").floor("h")


@st.cache_data(show_spinner="Reading the league sheet…", ttl=3600)
def _load_leaderboard(key):
    """The field's standings, or None if the sheet cannot be reached.

    Returning None rather than raising keeps a network blip from taking down a
    page whose real work — the recommendation — needs no network at all.
    """
    try:
        return leaderboard(load_field(), TEAM_WINS)
    except Exception:
        return None


@st.cache_data(show_spinner="Pricing every legal pick…")
def _options(used: frozenset, week: int, pool: int, blocked: tuple, key: tuple):
    return week_options(tw, set(used), week, pool_size=pool,
                        blocked=_as_blocked(blocked))


@st.cache_data(show_spinner=False)
def _plan(used: frozenset, week: int, blocked: tuple, key: tuple):
    return optimal_plan(tw, set(used), week, blocked=_as_blocked(blocked))


@st.cache_data(show_spinner=False)
def _plan_around(used: frozenset, week: int, team: str, blocked: tuple, key: tuple):
    """The rest of the season re-solved around a pick we have committed to.

    Needed because the recommendation and the plan below it have to describe the
    same season: once an entry is steered off the unconstrained optimum, the
    weeks after it change too, and showing the old plan would quietly contradict
    the pick sitting above it.
    """
    return optimal_plan(tw, set(used), week, forced=(week, team),
                        blocked=_as_blocked(blocked))


def _recommend(opts: pd.DataFrame, eid: str) -> pd.Series:
    """The row to put on the card: the best pick for winning the pot, not for
    reaching Week 18.

    Inside the survival budget, two things separate one safe team from another
    and the pool pays for both. `pot_ev` is the ground gained in the weeks you
    survive — a team nobody else has thins the field around you. `team_wins` is
    the ground lost at the end, because the pick is also a retirement: whatever
    you play is gone from the inventory the tiebreaker scores.

    The budget is `eid`'s own, not a house setting: a mulligan in hand buys the
    room to chase ground that an entry on its last life does not have.

    `opts` arrives sorted by season survival, so its top row is the pure
    survival answer, which is what this falls back to when the pot has not been
    priced or the floor rules everything out.
    """
    if "pot_ev" not in opts.columns or "team_wins" not in opts.columns:
        return opts.iloc[0]
    near = opts[(opts["season_survival"]
                 >= SURVIVAL_KEEP[eid] * opts["season_survival"].max())
                & (opts["win_prob"] >= AGGRESSION_FLOOR)]
    if near.empty:
        return opts.iloc[0]
    return near.loc[(near["pot_ev"] - TIEBREAK_WEIGHT * near["team_wins"]).idxmax()]


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
    "Entries still alive", min_value=2, max_value=1000, value=60, step=1,
    key="cs_pool",
    help="Drives the EV column. Pot share is what you are actually playing for, "
         "and how much being different is worth depends entirely on how many "
         "people you would be splitting with. Count the entries that can still "
         "win, not the entries that bought in — after Week 4 that is 60 of the "
         "original 87, the other 27 having been chopped."))
diverge = st.sidebar.toggle(
    "Split the two entries", value=False, key="cs_diverge",
    help="Keeps Blake off every team in Alaina's blueprint. Off by default: the "
         "hedge reads well but simulates badly, because it buys decorrelation by "
         "handicapping one entry all season.")

# ══════════════════════════════════════════════════════════════════════════════
# SEASON STATE
# ══════════════════════════════════════════════════════════════════════════════
picks = load_picks()
graded = grade_picks(picks, sched)

state = {}
for eid, name, *_ in ENTRIES:
    mine = graded[graded["entry"] == eid].sort_values("week")
    lost = mine[mine["result"] == "loss"]
    this_week = mine[mine["week"] == cur_week]
    # One loss is survivable — the mulligan is a do-over after your FIRST failure.
    # The second one is what chops you, so elimination is keyed on losses[1] and
    # a single loss only spends the do-over.
    state[eid] = {
        "name": name,
        "history": mine,
        "used": set(mine["team"]),
        # Teams unavailable when solving THIS week: everything already spent in
        # some other week. The current week's own pick is not a constraint on it.
        "spent_elsewhere": frozenset(mine[mine["week"] != cur_week]["team"]),
        "mulligan_week": int(lost["week"].iloc[0]) if len(lost) >= 1 else None,
        "out_week": int(lost["week"].iloc[1]) if len(lost) >= 2 else None,
        "locked": this_week.iloc[0] if not this_week.empty else None,
    }
    state[eid]["mulligan"] = state[eid]["mulligan_week"] is None

alive = [s for s in state.values()
         if s["out_week"] is None or s["out_week"] >= cur_week]

# ── This week at a glance ─────────────────────────────────────────────────────
def _clock(ts) -> str:
    """'Sun Sep 13, 1:00 PM ET'. Hand-built because %-I / %-d are not portable."""
    if ts is None:
        return "schedule not loaded"
    return (f"{ts:%a %b} {ts.day}, {(ts.hour % 12) or 12}:{ts.minute:02d} "
            f"{'AM' if ts.hour < 12 else 'PM'} ET")


# The league deadline is per pick — you have until your own team kicks off — so
# the week-level number is only the earliest a pick could possibly be due.
first_kick = week_deadline(sched, cur_week)

if cur_week <= COMPASSION_THROUGH:
    # The default is per entry: the rule double-defaults to the away team if you
    # have already spent the home side, so two entries can be owed two teams.
    _defaults = {s["name"]: compassion_default(sched, cur_week, used_teams=s["used"])
                 for s in state.values()}
    _uniq = set(_defaults.values())
    _miss_value = (next(iter(_uniq)) or "You're out") if len(_uniq) == 1 else "Differs"
    _who = ", ".join("{}: {}".format(n, t or "no team left")
                     for n, t in _defaults.items())
    _miss_sub = (f"The Compassion Clause assigns you the home team of this week's last "
                 f"game — {_who}. Once only, and it expires after "
                 f"Week {COMPASSION_THROUGH}.")
else:
    _miss_value = "You're out"
    _miss_sub = (f"Past Week {COMPASSION_THROUGH} a missed pick is an automatic loss and "
                 "the commissioner removes your best remaining team.")

_mull = ", ".join(f"{s['name']}: " + ("mulligan intact" if s["mulligan"]
                                      else f"used Wk {s['mulligan_week']}")
                  for s in state.values())

st.markdown(
    f'<div class="cs-band">'
    f'<div class="stat-card"><div class="label">Picking</div>'
    f'<div class="value">Week {cur_week} of {LAST_WEEK}</div>'
    f'<div class="sub">First kickoff {_clock(first_kick)} — but each pick is due '
    f'only at its own team\'s kickoff</div></div>'
    f'<div class="stat-card"><div class="label">Still alive</div>'
    f'<div class="value">{len(alive)} of {len(ENTRIES)}</div>'
    f'<div class="sub">{_mull or "both entries are out"}</div></div>'
    f'<div class="stat-card"><div class="label">If you forget to pick</div>'
    f'<div class="value">{_miss_value}</div>'
    f'<div class="sub">{_miss_sub}</div></div>'
    f'</div>', unsafe_allow_html=True)

st.warning(
    f"**Locking a pick here does not enter it.** Email the team to "
    f"**choppedfootball@gmail.com** before that team kicks off — that is the only "
    f"way the league accepts a pick. Changes are allowed right up until both the "
    f"old and the new team have kicked off.", icon="📧")

if cur_week != LIVE_WEEK:
    st.caption(f"👀 Looking ahead — the live week is **{LIVE_WEEK}**. "
               "Picks can only be locked for the live week.")

st.info(
    "**The rule that decides everything:** a team can only be used once all season, so "
    "every pick is really two decisions — who you play, and who you retire. **Cost** "
    "prices the first (full-season survival given up against the best plan left, so "
    "0.00 keeps the most in reserve) and **Burn** prices the second (the projected "
    "wins of the team you are spending, which is what the pot's tiebreaker is settled "
    "in). The recommendation is not the 0.00 pick: it is the best **EV** inside a "
    "survival budget — "
    + " and ".join(f"{SURVIVAL_KEEP[e]:.0%} for {n}" for e, n in ENTRIES)
    + ", set by who still has a mulligan — because reaching Week 18 in a crowd is "
    "not the same thing as winning.",
    icon="🧠",
)

# ══════════════════════════════════════════════════════════════════════════════
# THE TWO ENTRIES
# ══════════════════════════════════════════════════════════════════════════════
recs = {}

# Solved before anything is drawn, because priority order and display order are
# not the same: a locked pick claims its team first (it is a fact, not a
# suggestion), then Alaina, then Blake around whatever is left.
#
# Divergence is now OFF by default, and the two entries are allowed to play the
# same team when that is simply the best pick.
#
# The hedging argument is intuitive and wrong here. Blocking Blake from Alaina's
# blueprint does decorrelate them, but the pot is winner-take-all, so what it
# really does is hand one entry the optimal season and the other a handicapped
# one — for the whole year, not just the week of the collision. Simulated as the
# pot is actually won (both entries in the same pool, a win counted when either
# of them takes it), splitting loses at every gate tried: at the shipped 0.86 it
# is 3.4% against 8.9%, and it never once came out ahead.
#
# Two good entries beat one good entry and one compromised entry, because the
# tiebreaker pays for holding strong teams and the split spends exactly those.
# The toggle stays so the behaviour can be inspected, but it is no longer the
# recommendation. When divergence IS on it still covers the whole remaining
# season rather than one week, since a split that only covers Week N quietly
# collides again in Week N+1.
board = {}          # eid -> {"opts", "best", "plan"} or None
blocked_pairs = [(cur_week, s["locked"]["team"])
                 for s in state.values() if s["locked"] is not None]

for eid in PRIORITY:
    s = state[eid]
    if s["out_week"] is not None and s["out_week"] < cur_week:
        continue
    blk = tuple(blocked_pairs) if diverge else ()

    if s["locked"] is not None:
        plan = _plan(frozenset(s["used"]), cur_week + 1, blk, PROJ_KEY)
    else:
        opts = _options(s["spent_elsewhere"], cur_week, pool_size, blk, PROJ_KEY)
        if opts.empty:
            board[eid] = None
            s["blueprint"] = pd.DataFrame()
            continue
        opts = opts.assign(team_wins=opts["team"].map(TEAM_WINS))
        best = _recommend(opts, eid)
        plan = _plan_around(s["spent_elsewhere"], cur_week, best["team"], blk, PROJ_KEY)
        board[eid] = {"opts": opts, "best": best, "plan": plan}

    s["blueprint"] = plan
    if diverge and not plan.empty:
        blocked_pairs.extend((int(w), t) for w, t in zip(plan["week"], plan["team"]))

for col, (eid, name) in zip(st.columns(2), ENTRIES):
    s = state[eid]
    with col:
        st.markdown(f"### {name}")
        st.caption(
            f"**Playing for the pot.** Of the picks that keep "
            f"{SURVIVAL_KEEP[eid]:.0%} of the best season still available, the one "
            "that gains the most ground on the field — and retires the least "
            "useful team while doing it.")

        # ── Out of the pool ───────────────────────────────────────────────────
        if s["out_week"] is not None and s["out_week"] < cur_week:
            gone = s["history"][s["history"]["week"] == s["out_week"]].iloc[0]
            st.error(f"Chopped in Week {s['out_week']} — **{gone['team']}** lost to "
                     f"{gone['opponent']}, and the mulligan was already gone "
                     f"(Week {s['mulligan_week']}).", icon="💀")
            st.caption(f"{len(s['used'])} teams spent. Nothing left to decide, but the "
                       "log below still shows how the run went.")
            continue

        _mull_note = ("🛟 mulligan intact" if s["mulligan"] else
                      f"⚠️ mulligan spent in Week {s['mulligan_week']} — the next loss ends it")
        st.caption(f"{len(ALL_TEAMS) - len(s['used'])} teams still unspent · "
                   f"{len(s['used'])} used · {_mull_note}")

        # ── Already locked for this week ──────────────────────────────────────
        if s["locked"] is not None:
            lk = s["locked"]
            icon = RESULT_ICON.get(lk["result"], "⏳")
            score = (f'{int(lk["points"])}–{int(lk["opp_points"])}'
                     if lk["result"] != "pending" else
                     f'{"vs" if lk["is_home"] else "@"} {lk["opponent"]}')
            st.markdown(
                f'<div class="stat-card cs-pick">'
                f'<div class="label">Locked · Week {cur_week}</div>'
                f'<div style="font-size:1.8rem;font-weight:800;margin:6px 0 2px;">'
                f'{_crest(lk["team"], 34)} {lk["team"]} {icon}</div>'
                f'<div class="sub">{score}</div></div>', unsafe_allow_html=True)
            recs[name] = lk["team"]

            if lk["result"] == "pending":
                if st.button("Unlock this pick", key=f"cs_unlock_{eid}"):
                    clear_pick(cur_week, eid)
                    st.rerun()

            # A loss only ends the season when the mulligan was already gone, so
            # the forward plan is suppressed in exactly that case and not merely
            # because the pick above it lost.
            rest = pd.DataFrame() if s["out_week"] == cur_week else s["blueprint"]
            if not rest.empty:
                with st.expander(
                        f"Plan from Week {cur_week + 1} "
                        f"({survival_probability(rest, s['mulligan']):.1%} to reach "
                        f"Week {LAST_WEEK})"):
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
        solved = board.get(eid)
        if solved is None:
            st.warning("No legal picks left for this week.")
            continue
        opts, best = solved["opts"], solved["best"]
        recs[name] = best["team"]

        # Say out loud why this is not simply the safest team on the board,
        # whenever it isn't — an unexplained 74% next to a 82% reads as a bug.
        #
        # Both cards always carry exactly two facts, so the two boxes come out the
        # same height and the controls underneath them line up. The safe entry
        # solves first and so is almost always sitting on the season optimum at
        # zero cost; without a second fact of its own its card was permanently a
        # line shorter than the aggressive one.
        _cost = (f'costs {best["cost_vs_best"]:.2%} of the season'
                 if best["cost_vs_best"] > 0 else "no cost to the season plan")
        _why = ([_cost] if "pot_ev" not in opts.columns else
                [f'{best["pot_ev"]:.2f} claim on the pot',
                 f'{best["popularity"]:.0%} of the field is on it'])
        _toll = f'<div class="sub" style="margin-top:4px;">{" · ".join(_why)}</div>'
        st.markdown(
            f'<div class="stat-card cs-pick">'
            f'<div class="label">Recommended · Week {cur_week}</div>'
            f'<div style="font-size:1.8rem;font-weight:800;margin:6px 0 2px;">'
            f'{_crest(best["team"], 34)} {best["team"]}</div>'
            f'<div class="sub">{best["win_prob"]:.0%} to win '
            f'{"vs" if best["is_home"] else "@"} {best["opponent"]}</div>{_toll}</div>',
            unsafe_allow_html=True)

        # The deadline that actually applies to this pick, which is later than the
        # week's first kickoff whenever the recommendation is not in the early
        # window — worth knowing before assuming a Sunday 1pm cutoff.
        st.caption(f"📧 Email **{best['team']}** to choppedfootball@gmail.com by "
                   f"{_clock(week_deadline(sched, cur_week, team=best['team']))}.")

        # The whole point of solving the season instead of the week: some team
        # with a better number than the recommendation is sitting right there and
        # is being passed over on purpose, because it is the only good answer to a
        # thin week later on. Left unsaid that reads as the model being wrong, so
        # name the teams and the weeks they are being saved for.
        _later = {t: int(w) for w, t in zip(solved["plan"]["week"], solved["plan"]["team"])
                  if int(w) > cur_week}
        _held = [(r["team"], _later[r["team"]]) for _, r in opts.iterrows()
                 if r["team"] in _later and r["win_prob"] > best["win_prob"]]
        # Always printed, even when nothing is being saved, so one entry's held-back
        # note does not shove that entry's controls a line lower than the other's.
        st.caption("🔒 Held back: " + " · ".join(f"**{t}** for Wk {w}"
                                                 for t, w in _held[:4]) if _held
                   else "🔓 Nothing stronger is being saved — this is the top team left.")

        # The recommendation is a recommendation, not a lock — the selectbox
        # defaults to it but lets a gut call be recorded, because a pick made in
        # the app and not written down is the one that gets forgotten by Sunday.
        labels = {r["team"]: (f'{r["team"]} · {r["win_prob"]:.0%} '
                              f'{"vs" if r["is_home"] else "@"} {r["opponent"]} '
                              f'(cost {r["cost_vs_best"] * 100:.2f})')
                  for _, r in opts.iterrows()}
        # Defaults to the card above rather than the first row: `opts` is sorted
        # by survival, and the aggressive entry's recommendation deliberately is
        # not the top of that list.
        teams = list(labels)
        choice = st.selectbox("Pick to lock in", teams,
                              index=teams.index(best["team"]),
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
        show["Burn"] = show["team_wins"].round(1)
        st.dataframe(show[["Matchup", "Win %", "Cost", "Field %", "EV", "Burn"]],
                     hide_index=True,
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
                             help="Your claim on the pot relative to an average "
                                  "entrant, where 1.00 is that average. CHOPPED is "
                                  "winner-take-all, so this is not a share you would "
                                  "actually be paid — it is how much ground you gain "
                                  "on the field in the weeks you survive."),
                         "Burn": st.column_config.NumberColumn(
                             "Burn", format="%.1f",
                             help="Projected final wins of the team you would be "
                                  "retiring. The pot's tiebreaker is the average "
                                  "wins of the teams you have left, so the low "
                                  "numbers here are the cheap ones to spend."),
                     })

        # The EV leader is not always the pick, and an unexplained gap between
        # the top of this column and the card above it reads as a bug. Two
        # things can push the recommendation off it: the survival budget, and
        # the tiebreaker. Say which, but only when the gap is big enough to be
        # a real decision rather than a rounding step.
        EV_EDGE_MIN = 0.02
        ev_best = opts.loc[opts["pot_ev"].idxmax()]
        if (ev_best["team"] != best["team"]
                and ev_best["pot_ev"] - best["pot_ev"] >= EV_EDGE_MIN):
            _survival_gate = SURVIVAL_KEEP[eid] * opts["season_survival"].max()
            _reason = (f'it leaves only {ev_best["season_survival"]:.1%} of a season '
                       f'against {best["season_survival"]:.1%}, below the '
                       f'{SURVIVAL_KEEP[eid]:.0%} floor'
                       if ev_best["season_survival"] < _survival_gate else
                       f'it would retire a {ev_best["team_wins"]:.1f}-win team against '
                       f'{best["team_wins"]:.1f}, and the tiebreaker is paid out of '
                       f'what you have left')
            st.caption(
                f'⚖️ Highest **EV** this week is **{ev_best["team"]}** '
                f'({ev_best["pot_ev"]:.2f} vs {best["pot_ev"]:.2f}) at '
                f'{ev_best["win_prob"]:.0%} to win — passed over because {_reason}.')

        plan_df = solved["plan"]
        if not plan_df.empty:
            with st.expander(
                    f"Season blueprint to Week {LAST_WEEK} "
                    f"({survival_probability(plan_df, s['mulligan']):.1%} to get there"
                    + (" with the mulligan" if s["mulligan"] else " with no mulligan left")
                    + ")"):
                p = plan_df.copy()
                p["Matchup"] = p.apply(
                    lambda r: f'{r["team"]} {"vs" if r["is_home"] else "@"} {r["opponent"]}',
                    axis=1)
                p["Win %"] = (p["win_prob"] * 100).round(0).astype(int)
                st.dataframe(p[["week", "Matchup", "Win %"]].rename(columns={"week": "Wk"}),
                             hide_index=True, use_container_width=True)
                st.caption(
                    "Not a schedule to obey — it is the assignment of teams to weeks "
                    "that makes this week's pick worth taking, which is what stops a "
                    "greedy Week 1 from stranding you in Week 14. It is re-solved every "
                    "week against what you have actually spent."
                    + (" With *Split the two entries* on, this blueprint is also barred "
                       "from every team the other entry has reserved, all season."
                       if diverge else ""))

if len(recs) == 2 and len(set(recs.values())) == 1:
    st.info(
        f"Both entries are on **{list(recs.values())[0]}** — deliberately. One upset "
        "does take you both out, but simulating the pool both ways says that is the "
        "cheaper risk: forcing the second entry off the best team handicaps it for "
        "the rest of the season, and the tiebreaker is settled in exactly the teams "
        "a split makes you spend. Use *Split the two entries* in the sidebar if you "
        "would rather hedge."
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
               "per the league rule. Hit **Get This Week's Scores** in the sidebar once "
               "the games are over; it downloads the results and rebuilds the ratings. "
               "(*Reload From Disk* does not download anything.)")

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
# LEAGUE LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏅 League leaderboard")

_board = _load_leaderboard(_SHEET_TTL_KEY)
if _board is None:
    st.caption("Could not reach the league sheet, so the standings are hidden "
               "rather than shown stale. Everything above is computed locally "
               "and is unaffected.")
else:
    _alive = _board[_board["alive"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Still alive", f"{len(_alive)} of {len(_board)}")
    c2.metric("Mulligan intact", int(_alive["mulligan"].sum()))
    c3.metric("Chopped", len(_board) - len(_alive))

    st.caption(
        "Ranked the way the pot is settled: still alive first, then mulligan "
        "intact, then **reserve** — the average projected final wins of the "
        "teams you have *not* spent, which is the league's literal tiebreaker. "
        "Reserve reads backwards from instinct. Everyone alive has spent the "
        "same number of teams and the 32 win totals add to a fixed 272, so a "
        "high reserve means you have been winning with cheap teams. The entries "
        "that have already taken a loss mostly score better on it, because the "
        "spotless ones got spotless by spending the best teams on the board.")

    _show = _board.copy()
    _show.insert(0, "#", range(1, len(_show) + 1))
    _show["mulligan"] = _show["mulligan"].map({True: "🛟", False: "⚠️"})
    _show["reserve"] = _show["reserve"].round(2)
    _hide_out = st.checkbox("Hide chopped entries", value=True, key="cs_lb_alive")
    if _hide_out:
        _show = _show[_show["alive"]]
    st.dataframe(
        _show[["#", "player", "entry", "status", "mulligan", "used",
               "reserve", "burned"]],
        hide_index=True, use_container_width=True,
        column_config={
            "player": "Player", "entry": "Entry", "status": "Status",
            "mulligan": st.column_config.TextColumn(
                "Mull", help="🛟 intact · ⚠️ spent — the next loss ends it"),
            "used": st.column_config.NumberColumn("Used", help="Teams spent"),
            "reserve": st.column_config.NumberColumn(
                "Reserve", format="%.2f",
                help="Average projected final wins of the teams still unspent. "
                     "This is the tiebreaker the pot is decided on."),
            "burned": st.column_config.TextColumn("Teams spent"),
        })

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

st.markdown(f"""
**Surviving is not the same as winning, and this pool pays only for winning.**

Simulating the rest of the season against the field's actual remaining teams makes
the gap embarrassing. A pure survival-maximiser reaches Week 18 *more often than any
other rule tried* — and wins the pot least often of all of them. It gets there in a
crowd, and then loses the tiebreak. Both entries are now solved for the pot instead,
which is why neither card is necessarily sitting on the 0.00-cost pick any more.

Three things decide it, in this order.
""")

_a, _b = st.columns(2)
with _a:
    st.markdown(f"""
**1 · Don't go out. Budget what that costs.**

The **Cost** column is survival given up against the best remaining season — not
against this Sunday's biggest favourite. It already knows that using {_hoarded}
today is what leaves you with nothing in Week {_worst_week}, which is why a team with
a better number can sit on the board untouched and the **Season blueprint** names the
week it is being saved for.

Anything that keeps enough of the best season still available — {SURVIVAL_KEEP["blake"]:.0%}
for Blake, {SURVIVAL_KEEP["alaina"]:.0%} for Alaina, who has no mulligan left to spend —
and wins outright at least {AGGRESSION_FLOOR:.0%} of the time — is treated as equally
survivable, and the next two criteria pick between them. The discipline this asks for
is refusing an 82% week when the tool says 78%: that gap is the tool charging you for
the team you would have burned.

**2 · Be somewhere the field isn't.**

Surviving alongside 80% of the pool moves you nowhere; surviving a week that halves
the field is how a pot is won. The **EV** column is that number, 1.00 being the pool
average. Inside the budget, being different is nearly free, so take it.
""")
with _b:
    st.markdown(f"""
**3 · Every pick is also a retirement — spend the cheap teams.**

This is the part that was missing, and it is worth more than the other two combined.
The pot is never split: when more than one of you survives, it goes to the average
final wins of the teams you have **not** used. Everyone alive has spent the same
number of teams and the league's 272 wins are fixed, so holding the most at the end
is exactly burning the fewest along the way.

The **Burn** column is what a pick costs you there — the projected final wins of the
team you are retiring, so low is cheap. The ideal week is a mediocre team that
happens to be a big favourite, which is what makes the doormats ({_top_targets}) so
valuable: they let you cash in a 7-win team instead of an 11-win one. The premium
teams ({_top_premium}) are the answer to the thin weeks *and* to the tiebreak, which
is one reason to hoard them rather than two.

**And work backwards from Week {_worst_week}.** It is the thinnest week left — the
best team available is only {_worst_pct}. Whoever you are saving for a rainy day,
that is the day.

**Never spend the mulligan on forgetting.** It converts your first loss into a free
week, and it is most of the gap between Blake's odds here and Alaina's. Burning it on
an unsent email means the first genuine upset ends you.
""")

st.caption(
    "*Split the two entries* is **off** by default, so both entries can land on the "
    "same team when that is genuinely the best pick. It looks like a free hedge and "
    "is not: the pot is winner-take-all, so blocking the second entry does not buy "
    "two chances, it buys one good season and one handicapped one. Simulated as the "
    "pot is actually won, splitting lost at every setting tried. Switch it on and "
    "Blake is barred from every team Alaina has reserved **for the week she reserved "
    "it** — different seasons, not the same one with a swap, so a team can still "
    "appear on both in different weeks. Alaina solves first either way, because she "
    "has no mulligan left.")

with st.expander("The rules, as written"):
    st.markdown(f"""
**The mulligan is a do-over for your first failure — including a loss.** Incorrect
(your team lost) and invalid (late, a repeat team, or never sent) both count as
failures, and the *second* one chops you. That is why the survival numbers on this
page are quoted as "at most one loss" whenever the mulligan is still intact; a
clean-sweep number would be the wrong bar and roughly a fifth of the real answer.

**The Goofball Compassion Clause is a different thing.** It covers only a missed or
invalid pick, once, and only through Week {COMPASSION_THROUGH}: you get assigned the
home team of that week's last game — or the away team if you have already used the
home side, and an automatic loss if you have used both. From
**Week {COMPASSION_THROUGH + 1}** on, a missed pick is an automatic loss *and* the
commissioner takes your best remaining team, judged on record and point differential.
Double penalty: you are out, and on the way out you lose your answer to the thin weeks.

**Winner takes all — the pot is never split.** If several of you go out in the same
week it goes to a tiebreaker: combined wins of your **remaining** teams (a tie is half
a point) divided by how many teams you have left. This is not a footnote — simulated
out, it decides roughly half of the seasons in which anybody survives at all, because
the usual finish is two or three entries standing rather than one. Since everyone
still alive has spent the same number of teams, the divisor cancels and it reduces to
a single instruction: **burn the fewest wins.** That is the **Burn** column.

**Ties on the field count as wins** for both teams, so a pick only fails if your team
actually loses.

**Week 18 is not necessarily the end.** If more than one entrant is still standing
after the regular season, it continues into the playoffs — teams are *not* reloaded and
an unused mulligan carries forward. The plans on this page stop at Week
{LAST_WEEK} because that is where the schedule does, so treat a thin late-season
inventory as a real cost rather than a rounding error.

**Picks go by email, and only by email.** choppedfootball@gmail.com, before your team
kicks off. You may change a pick as long as neither the old nor the new team has
kicked off yet.
""")

st.caption(
    "You only need to outlast the other entrants, not the schedule — and with the "
    "mulligan intact you can be wrong once and still be standing."
)
