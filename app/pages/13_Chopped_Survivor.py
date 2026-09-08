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

# The three summary cards carry text of wildly different lengths ("2 of 2" next to
# a two-sentence Compassion Clause note), so left alone they end at three different
# heights. Streamlit already stretches the columns themselves to the tallest one —
# it is only the card inside that shrinks to its content, so growing the card to
# fill the slot is enough. Done in CSS rather than with a hard-coded min-height so
# it still holds when the text wraps to more lines on a narrow screen.
#
# Scoped to `.cs-fill` so this cannot reach the stat-cards on the other eight pages
# that share the class.
st.markdown("""
<style>
[data-testid="stElementContainer"]:has(.cs-fill) { height: 100%; }
[data-testid="stElementContainer"]:has(.cs-fill) [data-testid="stMarkdown"],
[data-testid="stElementContainer"]:has(.cs-fill) [data-testid="stMarkdownContainer"] {
    height: 100%;
}
.stat-card.cs-fill {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 0;
}
/* Streamlit gives every column a 320px min-width before it wraps, so on a laptop-
   width window three columns wrap 2-then-1: the third card lands full-width and
   short instead of matching the square pair above it. Equal height alone can't
   fix that — a card on its own row is a different shape no matter how tall it
   is. Forcing this specific row to stay in one line (and letting the columns
   shrink instead of wrap) keeps all three the same shape at any window width. */
div[data-testid="stHorizontalBlock"]:has(.cs-fill) {
    flex-wrap: nowrap !important;
}
div[data-testid="stHorizontalBlock"]:has(.cs-fill) > div[data-testid="stColumn"] {
    min-width: 0 !important;
    flex: 1 1 0 !important;
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

# Two entries into one pot are not two tickets unless they can fail separately,
# so each side is given a different job rather than the same optimiser twice.
# Edit the display names here; the ids are what the pick log is keyed on and
# changing one would orphan that entry's history.
ENTRIES = (
    ("blake", "Blake", "aggressive",
     "**Aggressive.** Same season-long plan, then the least-picked team among those "
     "still close to it — buying separation from the field with a little survival."),
    ("alaina", "Alaina", "safe",
     "**Safe.** The pick that maximises survival *to Week 18*, not this Sunday — "
     "teams are reserved for the weeks that need them."),
)
# Whose pick is solved first, which is not the order they are displayed in. The
# entry playing for the best chance to survive should never be the one pushed off
# the optimal team, so Alaina claims hers first and Blake diverges around it —
# which is what the aggressive side wants anyway.
PRIORITY = ("alaina", "blake")

# How far the aggressive entry will stray. Capped both ways on purpose: it may
# drop at most this much win probability below the safest team on the board, and
# never below the floor regardless. "Slightly aggressive" has to mean "takes the
# less popular of two close teams", not "takes an upset".
AGGRESSION_TOLERANCE = 0.08
AGGRESSION_FLOOR = 0.60

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


def _as_blocked(pairs: tuple) -> dict:
    """(week, team) pairs -> {week: {teams}}, the planner's blocking format.

    Kept as a flat tuple at the call sites because st.cache_data has to hash it
    and a dict of sets is not hashable.
    """
    out = {}
    for week, team in pairs:
        out.setdefault(week, set()).add(team)
    return out


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


def _recommend(opts: pd.DataFrame, mode: str) -> pd.Series:
    """The row to put on the card, given how much risk this entry is playing for.

    `opts` arrives sorted by season survival, so the safe answer is simply the
    top row. The aggressive answer is the biggest expected slice of the pot among
    the teams that are still close to it — surviving and winning are not the same
    thing in a pool, and a 71% team nobody else has thins the field in the weeks
    it comes in.
    """
    safe = opts.iloc[0]
    if mode != "aggressive" or "pot_ev" not in opts.columns:
        return safe
    near = opts[(opts["win_prob"] >= safe["win_prob"] - AGGRESSION_TOLERANCE)
                & (opts["win_prob"] >= AGGRESSION_FLOOR)]
    if near.empty:
        return safe
    return near.loc[near["pot_ev"].idxmax()]


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
    help="Both entries solve the same schedule, so on a lopsided week they can land "
         "on the same team — and get knocked out by the same upset. This keeps Blake "
         "off whatever Alaina is using.")

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

b1, b2, b3 = st.columns(3)
b1.markdown(
    f'<div class="stat-card cs-fill"><div class="label">Picking</div>'
    f'<div class="value">Week {cur_week} of {LAST_WEEK}</div>'
    f'<div class="sub">First kickoff {_clock(first_kick)} — but each pick is due '
    f'only at its own team\'s kickoff</div></div>', unsafe_allow_html=True)
b2.markdown(
    f'<div class="stat-card cs-fill"><div class="label">Still alive</div>'
    f'<div class="value">{len(alive)} of {len(ENTRIES)}</div>'
    f'<div class="sub">{_mull or "both entries are out"}</div></div>',
    unsafe_allow_html=True)
b3.markdown(
    f'<div class="stat-card cs-fill"><div class="label">If you forget to pick</div>'
    f'<div class="value">{_miss_value}</div>'
    f'<div class="sub">{_miss_sub}</div></div>', unsafe_allow_html=True)

st.warning(
    f"**Locking a pick here does not enter it.** Email the team to "
    f"**choppedfootball@gmail.com** before that team kicks off — that is the only "
    f"way the league accepts a pick. Changes are allowed right up until both the "
    f"old and the new team have kicked off.", icon="📧")

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

# Solved before anything is drawn, because priority order and display order are
# not the same: a locked pick claims its team first (it is a fact, not a
# suggestion), then Alaina, then Blake around whatever is left.
#
# Divergence is applied to the whole remaining season, not just to this week.
# Both entries solve the same schedule, so left alone their blueprints are
# near-identical — the same premium team earmarked for the same thin week — and
# a split that only covers Week N quietly collides again in Week N+1. Blocking
# each entry from the other's entire blueprint is what makes the second entry a
# genuinely different season rather than the first one with a swap.
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
        mode = next(m for i, _n, m, _s in ENTRIES if i == eid)
        opts = _options(s["spent_elsewhere"], cur_week, pool_size, blk, PROJ_KEY)
        if opts.empty:
            board[eid] = None
            s["blueprint"] = pd.DataFrame()
            continue
        best = _recommend(opts, mode)
        plan = _plan_around(s["spent_elsewhere"], cur_week, best["team"], blk, PROJ_KEY)
        board[eid] = {"opts": opts, "best": best, "plan": plan}

    s["blueprint"] = plan
    if diverge and not plan.empty:
        blocked_pairs.extend((int(w), t) for w, t in zip(plan["week"], plan["team"]))

for col, (eid, name, mode, style) in zip(st.columns(2), ENTRIES):
    s = state[eid]
    with col:
        st.markdown(f"### {name}")
        st.caption(style)

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
        if "pot_ev" not in opts.columns:
            _why = [_cost]
        elif mode == "aggressive":
            _why = [f'{best["pot_ev"]:.2f} claim on the pot',
                    f'only {best["popularity"]:.0%} of the field is on it']
        else:
            _why = [f'{best["popularity"]:.0%} of the field is on it', _cost]
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
                             help="Your claim on the pot relative to an average "
                                  "entrant, where 1.00 is that average. CHOPPED is "
                                  "winner-take-all, so this is not a share you would "
                                  "actually be paid — it is how much ground you gain "
                                  "on the field in the weeks you survive."),
                     })

        # Survival and pot share can disagree, and when they do it is worth
        # saying out loud rather than burying in a column — but it is a judgement
        # call about how much variance you want, so it is surfaced, not obeyed.
        # Only when the edge is real: the EV leader is often ahead by less than a
        # rounding step, and trading eight points of win probability for 0.6% of
        # the pot is not a decision worth putting in front of anyone.
        # Only for the safe entry: the aggressive one has already taken the EV
        # pick, so pointing at it there would just be reading its own card back.
        EV_EDGE_MIN = 0.02
        ev_best = opts.loc[opts["pot_ev"].idxmax()]
        if (mode == "safe" and ev_best["team"] != best["team"]
                and ev_best["pot_ev"] - best["pot_ev"] >= EV_EDGE_MIN):
            st.caption(
                f'⚖️ Highest **EV** this week is **{ev_best["team"]}** '
                f'({ev_best["pot_ev"]:.2f} vs {best["pot_ev"]:.2f}) — only '
                f'{ev_best["win_prob"]:.0%} to win, but just '
                f'{ev_best["popularity"]:.0%} of the field is on it, so it gains '
                f'the most ground when it hits. The pick above is still the '
                f'survival-maximising one.')

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
**Blake — press the edge**

Where two teams are close, take the one the pool is not on. The card only ever
strays inside guardrails: at most {AGGRESSION_TOLERANCE:.0%} of win probability
below the safest team on the board, and never below {AGGRESSION_FLOOR:.0%} to win
outright. That is deliberately narrow — this is not hunting upsets, it is refusing
to be on the same team as everybody else when the cost of being different is a
rounding error.

What it buys is the weeks you both survive. Surviving alongside 80% of the pool
moves you nowhere; surviving a week that halves the field is how a pot is actually
won. The **EV** column is that number: 1.00 is the pool average.

With *Split the two entries* on, this side is also barred from every team Alaina has
reserved **for the week she reserved it**, not just from this Sunday's pick. So the
two blueprints are genuinely different seasons rather than the same one with a swap
— and the same team can still appear on both, in different weeks.
""")
with _b:
    st.markdown(f"""
**Alaina — hold the safest road**

Take the 0.00-cost pick, and take it first: this side gets first claim on the board
and Blake works around it, so you are never the one pushed off a team you needed.

The thing worth understanding is that 0.00 is *not* the biggest favourite. Every
week is priced by solving all 18 weeks at once and assigning one team to each — so
the cost column already knows that using {_hoarded} today is what leaves you with
nothing in Week {_worst_week}. Taking the 0.00 pick every week is not eighteen
greedy decisions; it is one plan, re-solved each week against what you have actually
spent. That is why a team with a better number can sit on the board untouched, and
why the **Season blueprint** above shows which week it is being saved for.

The discipline this asks for is refusing an 82% week when the tool says 78%. That
gap is not the tool being wrong — it is the tool charging you for the team you would
have burned.
""")

st.markdown(f"""
**Both entries, same three habits**

**1 · Spend the cheap wins, hoard the expensive ones.** Almost every good week comes
from playing somebody against the same handful of bad teams — right now that is
{_top_targets}. The premium teams ({_top_premium}) are a limited resource. Burning one
for an 81% week when you could have had 78% from a team you will never want again is
how people lose this pool.

**2 · Work backwards from Week {_worst_week}.** It is the thinnest week left: the best
team available is only {_worst_pct}. Whoever you are saving for a rainy day, that
is the day. Do not arrive there holding only teams you were avoiding.

**3 · Never spend the mulligan on forgetting.** It is worth more than any single pick:
it converts your first loss into a free week. Burning it because an email did not get
sent means the first genuine upset ends you, and upsets are the one thing no plan
prevents.
""")

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
a point) divided by how many teams you have left. Hoarding strong teams both raises
your late-season floor and wins that tiebreak, so it is one plan, not two.

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
