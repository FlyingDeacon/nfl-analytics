import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils.styles import NFL_CSS, TEAM_COLORS
from utils.data_loader import load_teams, load_schedules, get_logo, get_base_dir, _file_mtime
from utils.survivor import MARKET_WEIGHT
from utils.projection import matchup_tables, input_paths
from utils.nav import render_sidebar_nav, render_last_updated
from utils.gate import require_passcode

st.set_page_config(page_title="Weekly Matchups · NFL", page_icon="🗓️", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

require_passcode("Weekly Matchups")

render_sidebar_nav(current_page="12_Weekly_Matchups")

if st.button("← Back to Season Projections", key="wm_back"):
    st.switch_page("pages/9_Record_Predictions.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">🗓️</div>
    <div>
        <div class="title">Weekly Matchups</div>
        <div class="subtitle">Every 2026 game, priced by the model and the sportsbook</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)
render_last_updated(*input_paths())

# ── Matchup card styling ─────────────────────────────────────────────────────
# auto-fill/minmax rather than st.columns: the slate is 13-16 games and the card
# count per row should follow the window, not a number fixed at render time.
st.markdown("""
<style>
.mu-grid { display: grid; gap: 14px; margin-top: 6px;
           grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); }
.mu-card { background: var(--glass); border: 1px solid var(--glass-border);
           border-radius: 16px; padding: 13px 16px 15px;
           box-shadow: var(--shadow-md); backdrop-filter: blur(12px);
           -webkit-backdrop-filter: blur(12px); }
.mu-head { display: flex; justify-content: space-between; align-items: center;
           margin-bottom: 12px; gap: 8px; }
.mu-kick { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.05em;
           text-transform: uppercase; font-weight: 600; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; }
.mu-badge { font-size: 0.64rem; font-weight: 800; padding: 3px 9px;
            border-radius: 999px; letter-spacing: 0.06em;
            text-transform: uppercase; color: #fff; white-space: nowrap; }
.mu-teams { display: flex; align-items: center; gap: 6px; }
.mu-side { flex: 1; display: flex; flex-direction: column; align-items: center;
           gap: 5px; min-width: 0; }
.mu-side img { width: 42px; height: 42px; object-fit: contain; }
.mu-noimg { width: 42px; height: 42px; display: flex; align-items: center;
            justify-content: center; font-weight: 800; color: var(--muted); }
.mu-abbr { font-size: 0.82rem; font-weight: 700; color: var(--text-sec);
           letter-spacing: 0.03em; }
.mu-pct { font-size: 1.4rem; font-weight: 800; line-height: 1; }
.mu-at { font-size: 0.72rem; font-weight: 700; color: var(--muted);
         letter-spacing: 0.06em; flex: 0 0 auto; padding-bottom: 22px; }
.mu-bar { display: flex; height: 7px; border-radius: 999px; overflow: hidden;
          margin: 14px 0 9px; }
.mu-foot { display: flex; justify-content: space-between; align-items: center;
           font-size: 0.72rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
_base = get_base_dir()
teams_df = load_teams(_mtime=_file_mtime(_base / "data/raw/teams.csv"))
schedules = load_schedules(_mtime=_file_mtime(_base / "data/raw/schedules.csv"))
blended, _ = matchup_tables()

# Kickoff day/time is not part of the projection, so pull it back off the
# schedule — it is what turns a row of numbers into a recognisable fixture.
_kick_cols = [c for c in ("weekday", "gametime") if c in schedules.columns]
if _kick_cols:
    _kick = schedules[(schedules["season"] == 2026) & (schedules["game_type"] == "REG")]
    blended = blended.merge(
        _kick[["week", "home_team", "away_team"] + _kick_cols].drop_duplicates(),
        on=["week", "home_team", "away_team"], how="left")

weeks = sorted(blended["week"].unique())
wk = st.selectbox("Week", weeks, key="wm_week", format_func=lambda w: f"Week {w}")

slate = blended[blended["week"] == wk].copy()
# Sort by how strong the favourite is, not by the home side's number — sorting
# on p_home interleaves road favourites with genuine coin flips.
slate["fav_p"] = slate["p_home_blend"].clip(lower=1 - slate["p_home_blend"])
slate = slate.sort_values("fav_p", ascending=False)


def _band(fav_p: float) -> tuple[str, str]:
    """Confidence label + colour. A near coin-flip is different information
    from a lock, and should not need a percentage comparison to spot."""
    if fav_p >= 0.75:
        return "Lock", "#15803d"
    if fav_p >= 0.65:
        return "Strong", "#4f46e5"
    if fav_p >= 0.55:
        return "Lean", "#b45309"
    return "Toss-up", "#8b8fa8"


def _kickoff(row) -> str:
    day = row.get("weekday") if "weekday" in row.index else None
    time = row.get("gametime") if "gametime" in row.index else None
    parts = []
    if isinstance(day, str) and day:
        parts.append(day[:3])
    if isinstance(time, str) and ":" in time:
        h, m = time.split(":")[:2]
        h = int(h)
        parts.append(f"{(h % 12) or 12}:{m} {'AM' if h < 12 else 'PM'}")
    return " · ".join(parts) if parts else f"Week {int(row['week'])}"


def _crest(abbr: str) -> str:
    url = get_logo(abbr, teams_df)
    return (f'<img src="{url}" alt="{abbr}">' if url
            else f'<div class="mu-noimg">{abbr}</div>')


def _card(row) -> str:
    ph = float(row["p_home_blend"])
    home, away = row["home_team"], row["away_team"]
    p_home, p_away = ph, 1 - ph
    fav_p = max(ph, 1 - ph)
    label, colour = _band(fav_p)

    # Colour each side with its own team, so the split bar reads as "this much
    # of the game belongs to that club" without needing a legend.
    c_home = TEAM_COLORS.get(home, "#4f46e5")
    c_away = TEAM_COLORS.get(away, "#8b8fa8")

    if row["has_market"]:
        mkt_home = float(row["p_home_mkt"])
        fav_mkt = mkt_home if ph >= 0.5 else 1 - mkt_home
        foot_right = f"Vegas {fav_mkt:.0%}"
    else:
        foot_right = "No line yet"

    def side(abbr: str, p: float, colour: str) -> str:
        # Mute the underdog so the favourite is the number the eye lands on.
        strong = p >= 0.5
        return (f'<div class="mu-side">{_crest(abbr)}'
                f'<div class="mu-abbr">{abbr}</div>'
                f'<div class="mu-pct" style="color:{colour if strong else "var(--muted)"};'
                f'opacity:{1 if strong else 0.75};">{p:.0%}</div></div>')

    return (
        f'<div class="mu-card">'
        f'<div class="mu-head"><span class="mu-kick">{_kickoff(row)}</span>'
        f'<span class="mu-badge" style="background:{colour};">{label}</span></div>'
        f'<div class="mu-teams">{side(away, p_away, c_away)}'
        f'<span class="mu-at">@</span>{side(home, p_home, c_home)}</div>'
        f'<div class="mu-bar">'
        f'<div style="width:{p_away:.4%};background:{c_away};"></div>'
        f'<div style="width:{p_home:.4%};background:{c_home};"></div></div>'
        f'<div class="mu-foot"><span>{away} at {home}</span>'
        f'<span>{foot_right}</span></div></div>'
    )


# ── Week summary ─────────────────────────────────────────────────────────────
_locks = int((slate["fav_p"] >= 0.75).sum())
_tossups = int((slate["fav_p"] < 0.55).sum())
_priced = int(slate["has_market"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", len(slate))
c2.metric("Locks (75%+)", _locks)
c3.metric("Toss-ups (<55%)", _tossups)
c4.metric("Vegas lines", f"{_priced}/{len(slate)}")

st.markdown('<div class="mu-grid">'
            + "".join(_card(r) for _, r in slate.iterrows())
            + "</div>", unsafe_allow_html=True)

st.caption(
    f"Where a sportsbook has posted a line the number is {MARKET_WEIGHT:.0%} market / "
    f"{1 - MARKET_WEIGHT:.0%} model; otherwise it is the model alone — the power-rating "
    "gap plus home field, read off a normal curve. Across the 67 priced 2026 games the "
    "two agree to within 3 points on average, which is why the later weeks are shown "
    "at all rather than left blank until the books open."
)

st.markdown("---")
if st.button("🔪 Open the CHOPPED Survivor planner", key="wm_to_chopped",
             use_container_width=True):
    st.switch_page("pages/13_Chopped_Survivor.py")
