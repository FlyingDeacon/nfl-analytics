from __future__ import annotations

import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import (
    load_weekly, load_teams, get_logo, get_base_dir, _file_mtime, _normalize_name,
)
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Player Comparison · NFL", page_icon="⚔️", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

render_sidebar_nav(current_page="11_Player_Comparison")

if st.button("← Back to Fantasy Football", key="pc_back_btn"):
    st.switch_page("pages/5_Fantasy.py")

st.markdown("""
<div class="nfl-page-header">
    <div class="icon">⚔️</div>
    <div>
        <div class="title">Player Comparison</div>
        <div class="subtitle">Head-to-head: model projections, ESPN consensus, and last season's real production</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE STYLES — the VS arena
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.pc-arena {
    display: grid;
    grid-template-columns: 1fr 96px 1fr;
    align-items: stretch;
    gap: 0;
    margin: 0.2rem 0 1.6rem;
}
.pc-side {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem 1.4rem 1.3rem;
    text-align: center;
    overflow: hidden;
    box-shadow: var(--shadow-md);
}
/* Team-tinted wash behind the hero */
.pc-side::before {
    content: "";
    position: absolute; inset: 0 0 auto 0;
    height: 132px;
    background: linear-gradient(180deg, var(--tc-wash), transparent);
    pointer-events: none;
}
.pc-side::after {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--tc), var(--tc-soft));
}
.pc-side.right::after {
    background: linear-gradient(270deg, var(--tc), var(--tc-soft));
}

/* ── Headshot bubble ── */
.pc-bubble { position: relative; width: 156px; height: 156px; margin: 0.2rem auto 0; }
.pc-ring {
    width: 156px; height: 156px; border-radius: 50%; padding: 4px;
    background: conic-gradient(from 200deg, var(--tc), var(--tc-soft), var(--tc));
    box-shadow: 0 14px 34px var(--tc-glow), 0 2px 6px rgba(0,0,0,0.10);
}
.pc-face {
    width: 100%; height: 100%; border-radius: 50%; overflow: hidden;
    background: radial-gradient(circle at 50% 30%, #ffffff, #e9ecf5);
    display: flex; align-items: flex-end; justify-content: center;
}
.pc-face img { width: 100%; height: 100%; object-fit: cover; object-position: top center; }
.pc-face .pc-initials {
    font-family: 'DM Sans', sans-serif; font-size: 2.4rem; font-weight: 800;
    color: var(--tc-txt); align-self: center; letter-spacing: -0.02em; opacity: 0.55;
}
.pc-teamchip {
    position: absolute; right: -4px; bottom: 4px;
    width: 50px; height: 50px; border-radius: 50%;
    background: #fff; border: 2px solid #fff; padding: 6px;
    object-fit: contain;
    box-shadow: 0 5px 16px rgba(0,0,0,0.20);
}

/* ── Identity ── */
.pc-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;
    color: var(--text); line-height: 1.15;
    margin: 0.95rem 0 0.1rem;
}
.pc-pills { display: flex; gap: 6px; justify-content: center; flex-wrap: wrap; margin-top: 8px; }
.pc-pill {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 11px; border-radius: 999px;
    background: var(--tc-txt); color: #fff;
    box-shadow: 0 2px 6px var(--tc-glow);
}
.pc-pill.ghost {
    background: var(--surface2); color: var(--text-sec);
    border: 1px solid var(--border); box-shadow: none;
}
.pc-pill.warn { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; box-shadow: none; }

/* ── Team widget ── */
.pc-teamcard {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    margin: 0.95rem auto 0; padding: 8px 14px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; width: fit-content; max-width: 100%;
}
.pc-teamcard img { width: 26px; height: 26px; object-fit: contain; }
.pc-teamcard .tn {
    font-size: 0.82rem; font-weight: 700; color: var(--text); line-height: 1.1;
}
.pc-teamcard .td {
    font-size: 0.66rem; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600;
}

/* ── Headline projection ── */
.pc-headline { margin-top: 1.1rem; }
.pc-headline .big {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
    color: var(--tc-txt); line-height: 1;
}
.pc-headline .cap {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted); font-weight: 700; margin-top: 6px;
}

/* ── Mini stat strip inside each side ── */
.pc-mini { display: flex; gap: 8px; margin-top: 1.1rem; }
.pc-mini > div {
    flex: 1; min-width: 0; padding: 9px 4px;
    background: var(--surface2); border: 1px solid var(--border); border-radius: 11px;
}
.pc-mini .v {
    font-family: 'DM Sans', sans-serif; font-size: 1.05rem; font-weight: 800;
    color: var(--text); line-height: 1;
}
.pc-mini .k {
    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); font-weight: 700; margin-top: 5px;
}

/* ── Center VS column ── */
.pc-center {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    position: relative;
}
.pc-center .line {
    flex: 1; width: 2px;
    background: linear-gradient(180deg, transparent, var(--border), transparent);
}
.pc-vs {
    width: 62px; height: 62px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Sans', sans-serif; font-size: 1.15rem; font-weight: 800;
    letter-spacing: 0.02em; color: #fff;
    background: linear-gradient(135deg, #1e1e2e, #4a4e69);
    box-shadow: 0 8px 24px rgba(0,0,0,0.22), 0 0 0 6px var(--bg);
    margin: 10px 0;
}

/* ── Verdict banner ── */
.pc-verdict {
    display: flex; align-items: center; justify-content: center; gap: 14px;
    padding: 0.85rem 1.2rem; margin-bottom: 1.4rem;
    background: var(--glass); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border); border-radius: 14px;
    box-shadow: var(--shadow-sm);
}
.pc-verdict .score {
    font-family: 'DM Sans', sans-serif; font-size: 1.5rem; font-weight: 800;
    line-height: 1;
}
.pc-verdict .txt { font-size: 0.85rem; color: var(--text-sec); font-weight: 500; }
.pc-verdict .txt b { color: var(--text); }

/* ── Metric comparison rows ── */
.pc-group { margin: 0 auto 1.5rem; max-width: 780px; }
.pc-group-title {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.14em;
    font-weight: 700; color: var(--muted);
    padding-bottom: 8px; margin-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
.pc-row {
    display: grid;
    grid-template-columns: 88px 1fr 88px;
    align-items: center;
    gap: 16px;
    padding: 11px 6px;
    border-radius: 10px;
    transition: background 0.15s ease;
}
.pc-row:hover { background: var(--surface); }
.pc-val {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.18rem; font-weight: 700; color: var(--text-sec);
    line-height: 1.1; white-space: nowrap;
}
.pc-val.left  { text-align: right; }
.pc-val.right { text-align: left; }
.pc-val.win { font-weight: 800; font-size: 1.28rem; color: var(--wc); }
.pc-val .tick { font-size: 0.7rem; opacity: 0.9; margin: 0 3px; }
.pc-mid { min-width: 0; }
.pc-label {
    text-align: center; font-size: 0.74rem; font-weight: 600;
    color: var(--muted); letter-spacing: 0.03em;
    margin-bottom: 7px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pc-bar {
    display: flex; height: 8px; border-radius: 999px; overflow: hidden;
    background: var(--surface2); gap: 2px;
}
.pc-bar span { display: block; height: 100%; border-radius: 999px; transition: width 0.3s ease; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .pc-arena { grid-template-columns: 1fr; gap: 0; }
    .pc-center { flex-direction: row; padding: 6px 0; }
    .pc-center .line { flex: 1; width: auto; height: 2px;
        background: linear-gradient(90deg, transparent, var(--border), transparent); }
    .pc-vs { margin: 0 10px; width: 52px; height: 52px; font-size: 1rem; }
    .pc-row { grid-template-columns: 64px 1fr 64px; gap: 8px; }
    .pc-val { font-size: 1rem; }
    .pc-val.win { font-size: 1.06rem; }
    .pc-label { font-size: 0.66rem; }
    .pc-headline .big { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BIG_BOARD_DIR = ROOT_DIR / "data" / "derived"
BUILD_SCRIPT = ROOT_DIR / "scripts" / "build_big_boards.py"
SCORING_FORMATS = ["PPR", "Half PPR", "Standard"]
LAST_SEASON = 2025

SCORING_COL = {"PPR": "fantasy_points_ppr", "Standard": "fantasy_points", "Half PPR": "__half"}


def _board_path(scoring: str) -> Path:
    return BIG_BOARD_DIR / f"big_board_{scoring.replace(' ', '_')}.parquet"


def _ensure_board(scoring: str) -> Path | None:
    """Return the big-board parquet, building it on demand if it's missing."""
    path = _board_path(scoring)
    if path.exists():
        return path
    with st.spinner(f"Building your {scoring} big board…"):
        try:
            subprocess.run(
                [sys.executable, str(BUILD_SCRIPT), "--scoring", scoring],
                cwd=str(ROOT_DIR), check=True, capture_output=True, timeout=300,
            )
        except Exception:
            return None
    return path if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_board(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("vor", ascending=False).reset_index(drop=True)
    df["model_rank"] = range(1, len(df) + 1)
    df["pos_rank"] = df.groupby("pos")["vor"].rank(ascending=False, method="first").astype(int)
    df["espn_rank"] = pd.to_numeric(df["espn_overall"], errors="coerce").rank(
        method="first", ascending=True
    ).astype("Int64")
    df["name_key"] = df["player"].map(_normalize_name)
    return df


@st.cache_data(show_spinner=False)
def _last_season_stats(scoring: str, mtime: float) -> pd.DataFrame:
    """Per-player 2025 production, consistency and finish ranks."""
    wk = load_weekly(_mtime=mtime)
    if wk.empty:
        return pd.DataFrame()
    df = wk[(wk["season"] == LAST_SEASON) & (wk["season_type"] == "REG")].copy()
    df = df[df["position"].isin(["QB", "RB", "WR", "TE", "K"])]
    if df.empty:
        return pd.DataFrame()

    col = SCORING_COL[scoring]
    if col == "__half":
        df["__half"] = (df["fantasy_points"] + df["fantasy_points_ppr"]) / 2

    df["__boom"] = (df[col] >= 20).astype(int)
    df["__bust"] = (df[col] < 5).astype(int)

    g = df.groupby("player_display_name")
    out = pd.DataFrame({
        "position":  g["position"].last(),
        "last_pts":  g[col].sum().round(1),
        "games":     g[col].count(),
        "std":       g[col].std(),
        "best":      g[col].max().round(1),
        "boom":      g["__boom"].sum(),
        "bust":      g["__bust"].sum(),
        "headshot":  g["headshot_url"].last(),
    }).reset_index().rename(columns={"player_display_name": "player"})

    out["last_ppg"] = (out["last_pts"] / out["games"].replace(0, np.nan)).round(1)

    # Consistency 0–100: how little a player's weekly output swings around his own
    # average (100 = identical every week). Needs a real sample to mean anything.
    cv = out["std"] / out["last_ppg"].replace(0, np.nan)
    out["consistency"] = (100 * (1 - cv)).clip(0, 100).round(0)
    out.loc[out["games"] < 4, "consistency"] = np.nan

    # Finish ranks — overall and within position, by total points scored.
    out["finish"] = out["last_pts"].rank(ascending=False, method="first").astype(int)
    out["pos_finish"] = out.groupby("position")["last_pts"].rank(
        ascending=False, method="first"
    ).astype(int)

    out["name_key"] = out["player"].map(_normalize_name)
    return out


@st.cache_data(show_spinner=False)
def _team_meta(mtime: float) -> dict:
    teams_df = load_teams(_mtime=mtime)
    meta = {}
    for _, r in teams_df.iterrows():
        abbr = str(r["team_abbr"])
        meta[abbr] = {
            "name": r.get("team_name", abbr),
            "div":  r.get("team_division", ""),
            "logo": get_logo(abbr, teams_df) or "",
        }
    return meta


@st.cache_data(show_spinner=False)
def _weekly_log(scoring: str, mtime: float) -> pd.DataFrame:
    """2025 week-by-week fantasy points, used for the trend chart."""
    wk = load_weekly(_mtime=mtime)
    if wk.empty:
        return pd.DataFrame()
    df = wk[(wk["season"] == LAST_SEASON) & (wk["season_type"] == "REG")].copy()
    col = SCORING_COL[scoring]
    if col == "__half":
        df["__half"] = (df["fantasy_points"] + df["fantasy_points_ppr"]) / 2
    return df[["player_display_name", "week", col]].rename(
        columns={"player_display_name": "player", col: "pts"}
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
sel_scoring = st.sidebar.radio("Scoring Format", SCORING_FORMATS, key="pc_scoring")
POSITIONS = ["All", "QB", "RB", "WR", "TE", "K", "DEF"]

board_path = _ensure_board(sel_scoring)
if board_path is None:
    st.error("Big board unavailable. Open the 2026 Fantasy Predictions page once to build it.")
    st.stop()

_weekly_mtime = _file_mtime(get_base_dir() / "data" / "raw" / "weekly.csv")
board = _load_board(str(board_path), _file_mtime(board_path))
last = _last_season_stats(sel_scoring, _weekly_mtime)
tmeta = _team_meta(_file_mtime(get_base_dir() / "data" / "raw" / "teams.csv"))

data = board.merge(
    last.drop(columns=["player", "position"], errors="ignore"), on="name_key", how="left"
)

# ── Player pickers ───────────────────────────────────────────────────────────
def _names_for(pos: str) -> list[str]:
    pool = data if pos == "All" else data[data["pos"] == pos]
    return pool.sort_values("model_rank")["player"].tolist()


RANK_BY_PLAYER = dict(zip(data["player"], data["model_rank"]))


def _label(name: str) -> str:
    return f"#{RANK_BY_PLAYER[name]} {name}"


pick1, pick_mid, pick2 = st.columns([5, 1, 5])
with pick1:
    pos1 = st.selectbox("Position A", POSITIONS, key="pc_pos1")
    names1 = _names_for(pos1)
with pick_mid:
    st.markdown(
        "<div style='text-align:center;padding-top:4.7rem;font-weight:800;"
        "color:var(--muted);letter-spacing:0.1em;font-size:0.8rem;'>VS</div>",
        unsafe_allow_html=True,
    )
with pick2:
    pos2 = st.selectbox("Position B", POSITIONS, key="pc_pos2")
    names2 = _names_for(pos2)

if not names1 or not names2:
    st.info("No players available for one of the selected positions.")
    st.stop()

with pick1:
    p1 = st.selectbox(
        "Player A", names1, index=0, format_func=_label, key=f"pc_p1_{pos1}"
    )
with pick2:
    p2 = st.selectbox(
        "Player B", names2, index=min(1, len(names2) - 1),
        format_func=_label, key=f"pc_p2_{pos2}",
    )

A = data[data["player"] == p1].iloc[0]
B = data[data["player"] == p2].iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _num(v):
    """Coerce a cell to float, or None when it's missing."""
    if v is None or pd.isna(v):
        return None
    return float(v)


def _text(v) -> str:
    """Trimmed string for a cell, empty when the cell is missing."""
    return "" if v is None or pd.isna(v) else str(v).strip()


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix_white(hex_color: str, weight: float) -> str:
    """Blend a hex colour toward white — `weight` is how much colour remains."""
    r, g, b = _rgb(hex_color)
    return "#%02x%02x%02x" % tuple(
        int(round(c * weight + 255 * (1 - weight))) for c in (r, g, b)
    )


def _rgba(hex_color: str, alpha: float) -> str:
    return "rgba(%d,%d,%d,%.2f)" % (*_rgb(hex_color), alpha)


def _readable(hex_color: str) -> str:
    """Darken a team colour until it reads cleanly as text on the light theme.

    Several team colours (LAR gold, PIT yellow, LV silver) are far too light
    for body text, so they get progressively dimmed toward black.
    """
    r, g, b = _rgb(hex_color)
    while (0.2126 * r + 0.7152 * g + 0.0722 * b) > 120:
        r, g, b = (int(c * 0.85) for c in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)


def _flat(html: str) -> str:
    """Collapse generated markup to a single line.

    Streamlit renders through CommonMark, where a blank (or whitespace-only)
    line closes an HTML block and the indented remainder becomes a code block.
    """
    return "".join(line.strip() for line in html.splitlines())


def _fmt_rank(v):
    return "—" if v is None else f"#{int(v)}"


def _fmt_int(v):
    return "—" if v is None else f"{int(round(v))}"


def _fmt_1(v):
    return "—" if v is None else f"{v:,.1f}"


def _fmt_2(v):
    return "—" if v is None else f"{v:.2f}"


def _fmt_pct(v):
    return "—" if v is None else f"{int(round(v))}"


# (label, column, higher_is_better, formatter)
GROUPS = [
    ("2026 Outlook", [
        ("My Model Rank",     "model_rank",    False, _fmt_rank),
        ("ESPN Rank",         "espn_rank",     False, _fmt_rank),
        ("Positional Rank",   "pos_rank",      False, _fmt_rank),
        ("Expected Points",   "predicted_pts", True,  _fmt_1),
        ("Projected PPG",     "pred_ppg",      True,  _fmt_2),
        ("Projected Games",   "proj_games",    True,  _fmt_1),
        ("Value Over Repl.",  "vor",           True,  _fmt_1),
    ]),
    ("2025 Production", [
        ("Overall Finish",    "finish",        False, _fmt_rank),
        ("Positional Finish", "pos_finish",    False, _fmt_rank),
        ("Total Points",      "last_pts",      True,  _fmt_1),
        ("Points Per Game",   "last_ppg",      True,  _fmt_1),
        ("Games Played",      "games",         True,  _fmt_int),
        ("Best Week",         "best",          True,  _fmt_1),
    ]),
    ("Week-to-Week Reliability", [
        ("Consistency Rating", "consistency",  True,  _fmt_pct),
        ("Boom Weeks (20+)",   "boom",         True,  _fmt_int),
        ("Bust Weeks (<5)",    "bust",         False, _fmt_int),
    ]),
]


def _split(a, b, higher_better):
    """Left/right bar widths (percent) plus which side wins the metric.

    A metric only has a winner when both players have the stat — a rookie or
    D/ST with no prior-season line shouldn't hand the other side a free win.
    """
    if a is None and b is None:
        return 50.0, 50.0, 0
    if a is None:
        return 0.0, 100.0, 0
    if b is None:
        return 100.0, 0.0, 0

    if higher_better:
        wa, wb = max(a, 0.0), max(b, 0.0)
    else:
        # Ranks and bust counts: smaller is better, so compare reciprocals.
        # The +1 keeps a zero from producing an infinitely wide bar.
        wa, wb = 1.0 / (max(a, 0.0) + 1), 1.0 / (max(b, 0.0) + 1)

    total = wa + wb
    la = 50.0 if total <= 0 else 100.0 * wa / total
    if a == b:
        winner = 0
    elif higher_better:
        winner = 1 if a > b else -1
    else:
        winner = 1 if a < b else -1
    return la, 100.0 - la, winner


LC = TEAM_COLORS.get(str(A["team"]), "#4f46e5")
RC = TEAM_COLORS.get(str(B["team"]), "#0f766e")
if LC.lower() == RC.lower():           # same team — keep the two sides distinguishable
    RC = "#0f172a"
LC_TXT, RC_TXT = _readable(LC), _readable(RC)

# Tally wins across every metric so the verdict banner has something to report.
wins_a = wins_b = 0
for _, metrics in GROUPS:
    for _, key, hib, _f in metrics:
        w = _split(_num(A.get(key)), _num(B.get(key)), hib)[2]
        wins_a += w == 1
        wins_b += w == -1


# ══════════════════════════════════════════════════════════════════════════════
# HERO — the VS arena
# ══════════════════════════════════════════════════════════════════════════════

def _side_html(row, color, side_cls) -> str:
    name = str(row["player"])
    team = str(row["team"])
    pos = str(row["pos"])
    meta = tmeta.get(team, {"name": team, "div": "", "logo": ""})

    shot = _text(row.get("headshot"))
    if shot.startswith("http"):
        face = f'<img src="{shot}" alt="">'
    else:
        initials = "".join(p[0] for p in name.split()[:2]).upper()
        face = f'<div class="pc-initials">{initials}</div>'

    chip = f'<img class="pc-teamchip" src="{meta["logo"]}" alt="">' if meta["logo"] else ""
    logo = f'<img src="{meta["logo"]}" alt="">' if meta["logo"] else ""

    pills = f'<span class="pc-pill">{pos}</span><span class="pc-pill ghost">{team}</span>'
    if _num(row.get("is_rookie")):
        pills += '<span class="pc-pill ghost">Rookie</span>'
    if _text(row.get("injury_risk")):
        pills += '<span class="pc-pill warn">Injury Risk</span>'

    proj = _num(row.get("predicted_pts"))
    mini = [
        (_fmt_rank(_num(row.get("model_rank"))), "My Rank"),
        (_fmt_rank(_num(row.get("espn_rank"))),  "ESPN"),
        (f'{pos}{int(row["pos_rank"])}',          "Pos Rank"),
        (_text(row.get("round_grade")) or "—",    "Round"),
    ]
    mini_html = "".join(
        f'<div><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in mini
    )

    style = (f"--tc:{color};--tc-soft:{_mix_white(color, 0.28)};"
             f"--tc-wash:{_rgba(color, 0.18)};--tc-glow:{_rgba(color, 0.34)};"
             f"--tc-txt:{_readable(color)};")

    return f"""
    <div class="pc-side {side_cls}" style="{style}">
        <div class="pc-bubble">
            <div class="pc-ring"><div class="pc-face">{face}</div></div>
            {chip}
        </div>
        <div class="pc-name">{name}</div>
        <div class="pc-pills">{pills}</div>
        <div class="pc-teamcard">
            {logo}
            <div style="text-align:left;">
                <div class="tn">{meta['name']}</div>
                <div class="td">{_text(meta['div'])}</div>
            </div>
        </div>
        <div class="pc-headline">
            <div class="big">{_fmt_1(proj)}</div>
            <div class="cap">2026 Expected Points</div>
        </div>
        <div class="pc-mini">{mini_html}</div>
    </div>"""


if wins_a > wins_b:
    verdict = f'<b>{p1}</b> wins {wins_a} of {wins_a + wins_b} contested metrics'
elif wins_b > wins_a:
    verdict = f'<b>{p2}</b> wins {wins_b} of {wins_a + wins_b} contested metrics'
else:
    verdict = f'Dead even — <b>{wins_a}</b> metrics apiece'

st.markdown(_flat(f"""
<div class="pc-verdict">
    <span class="score" style="color:{LC_TXT};">{wins_a}</span>
    <span class="txt">{verdict}</span>
    <span class="score" style="color:{RC_TXT};">{wins_b}</span>
</div>
<div class="pc-arena">
    {_side_html(A, LC, "left")}
    <div class="pc-center">
        <div class="line"></div>
        <div class="pc-vs">VS</div>
        <div class="line"></div>
    </div>
    {_side_html(B, RC, "right")}
</div>
"""), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC ROWS
# ══════════════════════════════════════════════════════════════════════════════

rows_html = ""
for group_name, metrics in GROUPS:
    rows_html += f'<div class="pc-group"><div class="pc-group-title">{group_name}</div>'
    for label, key, hib, fmt in metrics:
        a, b = _num(A.get(key)), _num(B.get(key))
        la, lb, winner = _split(a, b, hib)
        a_cls = "pc-val left win" if winner == 1 else "pc-val left"
        b_cls = "pc-val right win" if winner == -1 else "pc-val right"
        a_tick = '<span class="tick">◀</span>' if winner == 1 else ""
        b_tick = '<span class="tick">▶</span>' if winner == -1 else ""
        rows_html += f"""
        <div class="pc-row">
            <div class="{a_cls}" style="--wc:{LC_TXT};">{fmt(a)}{a_tick}</div>
            <div class="pc-mid">
                <div class="pc-label">{label}</div>
                <div class="pc-bar">
                    <span style="width:{la:.1f}%;background:{LC};opacity:{1 if winner == 1 else 0.45};"></span>
                    <span style="width:{lb:.1f}%;background:{RC};opacity:{1 if winner == -1 else 0.45};"></span>
                </div>
            </div>
            <div class="{b_cls}" style="--wc:{RC_TXT};">{b_tick}{fmt(b)}</div>
        </div>"""
    rows_html += "</div>"

st.markdown(_flat(rows_html), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RADAR — percentile profile against the rest of the board
# ══════════════════════════════════════════════════════════════════════════════

RADAR_AXES = [
    ("Projection",  "predicted_pts", True),
    ("Efficiency",  "pred_ppg",      True),
    ("Availability", "proj_games",   True),
    ("Scarcity Value", "vor",        True),
    ("Consistency", "consistency",   True),
    ("Upside",      "best",          True),
]


def _percentile(key: str, value) -> float:
    """Where a player's value sits among everyone at his position (0–100)."""
    if value is None:
        return 0.0
    pool_vals = pd.to_numeric(data.loc[data["pos"].isin([A["pos"], B["pos"]]), key],
                              errors="coerce").dropna()
    if pool_vals.empty:
        return 0.0
    return float((pool_vals <= value).mean() * 100)


left_r = [_percentile(k, _num(A.get(k))) for _, k, _hb in RADAR_AXES]
right_r = [_percentile(k, _num(B.get(k))) for _, k, _hb in RADAR_AXES]
axis_labels = [lbl for lbl, _k, _hb in RADAR_AXES]

st.markdown("#### Player Profile")
st.caption("Percentile rank against every other player at these positions — further out is better.")

radar = go.Figure()
for vals, nm, color in ((left_r, p1, LC), (right_r, p2, RC)):
    radar.add_trace(go.Scatterpolar(
        r=vals + vals[:1],
        theta=axis_labels + axis_labels[:1],
        fill="toself",
        name=nm,
        line=dict(color=color, width=2.5),
        fillcolor=_rgba(color, 0.20),
        hovertemplate=f"<b>{nm}</b><br>%{{theta}}: %{{r:.0f}}th pct<extra></extra>",
    ))
radar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=PLOTLY_LAYOUT["font"],
    hoverlabel=PLOTLY_LAYOUT["hoverlabel"],
    margin=dict(l=60, r=60, t=40, b=40),
    height=430,
    polar=dict(
        bgcolor="rgba(255,255,255,0.55)",
        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e2e5ef",
                        tickvals=[25, 50, 75, 100], ticksuffix="",
                        tickfont=dict(size=9, color="#a8adc0"), angle=90, tickangle=0),
        angularaxis=dict(gridcolor="#e2e5ef",
                         tickfont=dict(size=11, color="#4a4e69")),
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# WEEK-TO-WEEK TREND
# ══════════════════════════════════════════════════════════════════════════════

log = _weekly_log(sel_scoring, _weekly_mtime)
log_a = log[log["player"] == p1].sort_values("week") if not log.empty else pd.DataFrame()
log_b = log[log["player"] == p2].sort_values("week") if not log.empty else pd.DataFrame()

if not log_a.empty or not log_b.empty:
    st.markdown(f"#### {LAST_SEASON} Week-by-Week")
    st.caption("Flatter lines mean a more predictable weekly starter; deep valleys are the busts.")

    trend = go.Figure()
    for lg, nm, color in ((log_a, p1, LC), (log_b, p2, RC)):
        if lg.empty:
            continue
        trend.add_trace(go.Scatter(
            x=lg["week"], y=lg["pts"].round(1),
            mode="lines+markers", name=nm,
            line=dict(color=color, width=2.6, shape="spline", smoothing=0.6),
            marker=dict(size=7, color=color, line=dict(color="#fff", width=1.5)),
            hovertemplate=f"<b>{nm}</b><br>Week %{{x}}: %{{y:.1f}} pts<extra></extra>",
        ))
        avg = lg["pts"].mean()
        trend.add_hline(y=avg, line=dict(color=color, width=1, dash="dot"), opacity=0.5)

    trend.update_layout(
        **PLOTLY_LAYOUT,
        title="",
        height=380,
        xaxis_title="Week", yaxis_title=f"Fantasy Points ({sel_scoring})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    st.plotly_chart(trend, use_container_width=True)
