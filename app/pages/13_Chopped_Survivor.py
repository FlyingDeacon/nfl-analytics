import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from utils.styles import NFL_CSS
from utils.data_loader import load_teams, get_logo, get_base_dir, _file_mtime
from utils.survivor import optimal_plan, week_options, survival_probability
from utils.projection import matchup_tables, input_paths
from utils.nav import render_sidebar_nav, render_last_updated
from utils.gate import require_passcode

st.set_page_config(page_title="CHOPPED Survivor · NFL", page_icon="🔪", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

require_passcode("CHOPPED Survivor")

render_sidebar_nav(current_page="13_Chopped_Survivor")

if st.button("← Back to Season Projections", key="cs_back"):
    st.switch_page("pages/9_Record_Predictions.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🔪</div>
    <div>
        <div class="title">CHOPPED Survivor</div>
        <div class="subtitle">One pick a week · no team twice · one mulligan · last entrant takes the pot</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)
render_last_updated(*input_paths())

_base = get_base_dir()
teams_df = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))
_, tw = matchup_tables()
_TW_KEY = _file_mtime(_base / "data/raw/schedules.csv")


def _crest(abbr: str, size: int = 26) -> str:
    url = get_logo(abbr, teams_df)
    return (f'<img src="{url}" width="{size}" style="vertical-align:middle;">'
            if url else f"<b>{abbr}</b>")


@st.cache_data(show_spinner="Optimising the rest of the season…")
def _plan(_used: frozenset, _week: int, _pool: int, _key: float):
    used = set(_used)
    return optimal_plan(tw, used, _week), week_options(tw, used, _week, pool_size=_pool)


@st.cache_data(show_spinner=False)
def _plan_from(_used: frozenset, _week: int, _team: str, _key: float):
    """The rest of the season re-solved around a pick we have committed to.

    Needed because the recommendation and the plan below it have to describe the
    same season: once an entry is steered off the unconstrained optimum, the
    weeks after it change too, and showing the old plan would quietly contradict
    the pick sitting above it.
    """
    return optimal_plan(tw, set(_used), _week, forced=(_week, _team))


st.info(
    "**The rule that decides everything:** a team can only be used once all season. "
    "So the best pick this week is rarely the biggest favourite — it is the one that "
    "leaves the strongest set of teams for the weeks still to come. Every option below "
    "is priced that way: **Cost** is how much full-season survival you give up versus "
    "the optimal choice, so a 0.00 cost is the pick that keeps the most value in reserve.",
    icon="🧠",
)

_wk_col, _pool_col, _div_col = st.columns([1, 1, 2])
cur_week = _wk_col.number_input("Current week", min_value=1, max_value=18, value=1,
                                step=1, key="cs_week")
pool_size = _pool_col.number_input(
    "Entries in the pool", min_value=2, max_value=1000, value=50, step=1,
    key="cs_pool",
    help="Drives the EV column. Pot share is what you are actually playing for, "
         "and how much being different is worth depends entirely on how many "
         "people you would be splitting with.")
diverge = _div_col.toggle(
    "Keep the two entries on different teams", value=True, key="cs_diverge",
    help="Both entries solve the same schedule, so left alone they converge on the "
         "same answer — and get knocked out by the same upset. This steers the second "
         "entry to its best pick that the first is not using.")

# Two entries into one pot are not two tickets unless they can fail separately,
# so each side is given a different job rather than the same optimiser twice.
STYLES = ("**Chalk.** Take the mathematically best pick every week.",
          "**Hedge.** Take the best team the other entry is not using.")

all_teams = sorted(tw["team"].unique())
entries = st.columns(2)
recs = {}
taken = None      # entry 1's pick, so entry 2 can be steered off it

for i, (col, label) in enumerate(zip(entries, ("Your entry", "Your wife's entry"))):
    with col:
        st.markdown(f"**{label}**")
        st.caption(STYLES[i])
        used = st.multiselect("Teams already used", all_teams, default=[],
                              key=f"cs_used_{i}")
        _, opts = _plan(frozenset(used), int(cur_week), int(pool_size), _TW_KEY)
        if diverge and taken is not None:
            opts = opts[opts["team"] != taken].reset_index(drop=True)
        if opts.empty:
            st.warning("No legal picks left for this week.")
            continue

        best = opts.iloc[0]
        recs[label] = best["team"]
        if taken is None:
            taken = best["team"]
        # Re-solve the remaining weeks around the pick actually being recommended,
        # which for the hedged entry is not the unconstrained optimum.
        plan_df = _plan_from(frozenset(used), int(cur_week), best["team"], _TW_KEY)

        _toll = (f'<div class="sub" style="margin-top:4px;">Costs {best["cost_vs_best"]:.2%} '
                 f'of season survival to stay off {taken}</div>'
                 if diverge and i == 1 and best["cost_vs_best"] > 0 else "")
        st.markdown(
            f'<div class="stat-card" style="text-align:center;padding:16px 12px;">'
            f'<div class="label">Recommended pick · Week {int(cur_week)}</div>'
            f'<div style="font-size:1.8rem;font-weight:800;margin:6px 0 2px;">'
            f'{_crest(best["team"], 34)} {best["team"]}</div>'
            f'<div class="sub">{best["win_prob"]:.0%} to win '
            f'{"vs" if best["is_home"] else "@"} {best["opponent"]}</div>{_toll}</div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
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

        with st.expander(f"Full plan to Week 18 ({survival_probability(plan_df):.1%} to run the table)"):
            p = plan_df.copy()
            p["Matchup"] = p.apply(
                lambda r: f'{r["team"]} {"vs" if r["is_home"] else "@"} {r["opponent"]}', axis=1)
            p["Win %"] = (p["win_prob"] * 100).round(0).astype(int)
            st.dataframe(p[["week", "Matchup", "Win %"]].rename(columns={"week": "Wk"}),
                         hide_index=True, use_container_width=True)
            st.caption(
                "This entry's own best path from here, so the two plans will show the "
                "same team in some later week. That is fine — divergence is a weekly "
                "decision, not a fixed schedule. Come back each week with the used "
                "teams updated and the toggle splits the live picks again.")

if len(recs) == 2 and len(set(recs.values())) == 1:
    st.warning(
        f"Both entries are on **{list(recs.values())[0]}**. Only one of you can win the pot, "
        "so playing the same team means one upset eliminates you both. Turn on "
        "*Keep the two entries on different teams* above to split them."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SEASON MAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🗺️ Season map")
st.caption("Recomputed from the current projection, so it stays true as the season moves.")

_rest = tw[tw["week"] >= int(cur_week)]

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

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**Danger weeks**")
    st.caption("Best pick available is weakest here")
    st.dataframe(pd.DataFrame({"Week": _danger.index, "Best available": _danger.values})
                 .assign(**{"Best available": lambda d: d["Best available"].astype(str) + "%"}),
                 hide_index=True, use_container_width=True)
with m2:
    st.markdown("**Premium teams**")
    st.caption("Weeks left where they are 70%+ favourites")
    st.dataframe(pd.DataFrame({"Team": _premium.index, "Good weeks": _premium.values}),
                 hide_index=True, use_container_width=True)
with m3:
    st.markdown("**Teams to attack**")
    st.caption("Weeks left where they are 30%-or-worse dogs")
    st.dataframe(pd.DataFrame({"Team": _targets.index, "Bad weeks": _targets.values}),
                 hide_index=True, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 How to play it")

_worst_week = int(_danger.index[0])
_top_premium = ", ".join(f"{t} ({n})" for t, n in _premium.head(4).items())
_top_targets = ", ".join(f"{t} ({n})" for t, n in _targets.head(4).items())

_a, _b = st.columns(2)
with _a:
    st.markdown(f"""
**Your entry — play the chalk**

Take the 0.00-cost pick every single week, no exceptions. This is the highest
survival path that exists given the schedule, and its whole value is that it never
gets clever. You are not trying to be different from the pool here; you are trying
to still be alive in Week {_worst_week}, and most entrants will not be, because they
will have spent {_premium.index[0]} on an early blowout they did not need.

The discipline this asks for is refusing an 82% week when the tool says 78%. That
gap is not the tool being wrong — it is the tool charging you for the team you would
have burned.
""")
with _b:
    st.markdown(f"""
**Her entry — play the hedge**

Take the best team your entry is not using. It will usually cost a fraction of a
percent this week, and it buys the only thing that matters with two entries: you
cannot both die to the same upset. If the chalk pick goes down on a last-second
field goal, one of you is still standing.

Over the season this naturally builds a different inventory of remaining teams, so
by Week 8 the two entries are not near-copies with one swap — they are genuinely
covering different outcomes.
""")

st.markdown(f"""
**Both entries, same five rules**

**1 · Spend the cheap wins, hoard the expensive ones.** Almost every good week comes
from playing somebody against the same handful of bad teams — right now that is
{_top_targets}. The premium teams ({_top_premium}) are a limited resource. Burning one
for an 81% week when you could have had 78% from a team you will never want again is
how people lose this pool.

**2 · Work backwards from Week {_worst_week}.** It is the thinnest week left: the best
team available is only {_danger.iloc[0]}%. Whoever you are saving for a rainy day, that
is the day. Do not arrive there holding only teams you were avoiding.

**3 · Respect the two rule cliffs.**
Through **Week 5** the Goofball Compassion Clause covers one missed pick, defaulting to
the home team of that week's last game. It is one-time-only and it expires — treat it as
insurance, not as permission to forget. From **Week 6** on, a missed or invalid pick is an
automatic loss *and* the commissioner removes your best remaining team. That is a double
penalty: you burn the mulligan and lose your answer to the thin weeks at the same time.

**4 · The tiebreaker rewards the same discipline.** Ties are broken by the combined wins of
your **remaining** teams divided by how many you have left. Hoarding strong teams both raises
your late-season floor and wins ties, so there is no tension between playing to survive and
playing to win the tiebreak — it is one plan, not two.

**5 · Ties on the field count as wins.** A pick only fails if your team actually loses,
which slightly favours taking a road favourite over passing on a week.
""")

st.caption(
    "Survival percentages look brutal because running 18 straight weeks is genuinely hard — "
    "but every entrant faces the same gauntlet, and you only need to outlast them, not the "
    "schedule. The mulligan roughly doubles the numbers shown above."
)
