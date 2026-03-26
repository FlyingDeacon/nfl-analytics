import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.styles import NFL_CSS, TEAM_COLORS, PLOTLY_LAYOUT
from utils.data_loader import load_ratings, load_teams, get_logo, add_ranks
from utils.nav import render_sidebar_nav

st.set_page_config(page_title="Team Ratings · NFL", page_icon="📊", layout="wide")
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="1_Team_Ratings")

# ── Page header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nfl-page-header">
    <div class="icon">📊</div>
    <div>
        <div class="title">Team Ratings</div>
        <div class="subtitle">Points per game · defensive efficiency · net rating</div>
    </div>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Scroll preservation (prevents page jumping to top on re-run) ─────────────
components.html("""
<script>
(function() {
    var key = '_nfl_scroll_y';
    var p   = window.parent;
    var saved = sessionStorage.getItem(key);
    if (saved !== null) { p.scrollTo(0, +saved); }
    p.addEventListener('scroll', function() {
        sessionStorage.setItem(key, p.scrollY);
    }, { passive: true });
})();
</script>
""", height=0)

# ── Load data ────────────────────────────────────────────────────────────────
ratings = load_ratings()
teams   = load_teams()

# ── Sidebar filters ──────────────────────────────────────────────────────────
if "tr_v" not in st.session_state:
    st.session_state["tr_v"] = 0
_v = st.session_state["tr_v"]

seasons = sorted(ratings["season"].dropna().unique(), reverse=True)
selected_season = st.sidebar.selectbox("Season", seasons, key=f"tr_season_{_v}")

full_df = ratings[ratings["season"] == selected_season].copy()
full_df = add_ranks(full_df)

team_list = ["All Teams"] + sorted(full_df["team"].unique().tolist())
selected_team = st.sidebar.selectbox("Team", team_list, key=f"tr_team_{_v}")

sort_metric = st.sidebar.selectbox("Sort by", ["net_ppg", "ppg", "oppg", "games"], key=f"tr_sort_{_v}")
ascending   = sort_metric == "oppg"

if st.sidebar.button("Reset Filters", key="tr_reset", use_container_width=True):
    st.session_state["tr_v"] = _v + 1
    st.rerun()

# ── Apply filters ────────────────────────────────────────────────────────────
view_df = full_df.copy()
if selected_team != "All Teams":
    view_df = view_df[view_df["team"] == selected_team]

view_df = view_df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)

# ── Snapshot cards ───────────────────────────────────────────────────────────
st.markdown("### Snapshot")

if not full_df.empty:
    def logo_img(abbr):
        url = get_logo(abbr, teams)
        if url:
            return f'<img src="{url}" width="48" style="margin:6px 0;">'
        return ""

    c1, c2, c3 = st.columns(3)

    if selected_team != "All Teams" and not view_df.empty:
        # Show the selected team's own stats across all three cards
        row = view_df.iloc[0]
        cards = [
            (c1, "Offensive PPG",  row, "ppg",     "PPG",     "offense_rank"),
            (c2, "Defensive PPG",  row, "oppg",    "OPPG",    "defense_rank"),
            (c3, "Net Rating",     row, "net_ppg", "Net PPG", "overall_rank"),
        ]
    else:
        # League leaders for the season
        best_off = full_df.loc[full_df["ppg"].idxmax()]
        best_def = full_df.loc[full_df["oppg"].idxmin()]
        best_net = full_df.loc[full_df["net_ppg"].idxmax()]
        cards = [
            (c1, "Best Offense",    best_off, "ppg",     "PPG",      "offense_rank"),
            (c2, "Best Defense",    best_def, "oppg",    "OPPG",     "defense_rank"),
            (c3, "Best Net Rating", best_net, "net_ppg", "Net PPG",  "overall_rank"),
        ]

    for col, title, row, metric, unit, rank_col in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">{title}</div>
                {logo_img(row["team"])}
                <div class="value">{row["team"]}</div>
                <div class="sub">{row[metric]:.1f} {unit} &nbsp;·&nbsp; Rank #{int(row[rank_col])}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Team Profile launcher — always visible ───────────────────────────────────
with st.container():
    st.markdown("#### 🏟️ Team Profile")
    tp_col1, tp_col2 = st.columns([2, 1])
    with tp_col1:
        profile_team_options = sorted(full_df["team"].unique().tolist())
        default_profile = (
            selected_team if selected_team != "All Teams"
            else st.session_state.get("profile_team", profile_team_options[0])
        )
        if default_profile not in profile_team_options:
            default_profile = profile_team_options[0]
        profile_team_pick = st.selectbox(
            "Select a team to view full profile",
            profile_team_options,
            index=profile_team_options.index(default_profile),
            key="tr_profile_pick",
            label_visibility="collapsed",
        )
    with tp_col2:
        if st.button("View Team Profile →", key="goto_profile",
                     use_container_width=True, type="primary"):
            st.session_state["profile_team"] = profile_team_pick
            st.switch_page("pages/8_Team_Profile.py")

st.markdown("---")

# ── Table + Bar chart ────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1])

with left:
    st.markdown("#### Rankings Table")
    disp = view_df[["team", "overall_rank", "offense_rank", "defense_rank",
                    "games", "ppg", "oppg", "net_ppg"]].copy()
    for c in ["ppg", "oppg", "net_ppg"]:
        disp[c] = disp[c].round(1)
    disp.columns = ["Team", "Overall", "Off. Rank", "Def. Rank",
                    "Games", "PPG", "OPPG", "Net PPG"]
    st.dataframe(disp, hide_index=True, use_container_width=True)

with right:
    st.markdown("#### Net Rating")
    if not view_df.empty:
        bar_df = view_df.sort_values("net_ppg", ascending=False)
        colors = [TEAM_COLORS.get(t, "#4f46e5") for t in bar_df["team"]]
        fig = go.Figure(go.Bar(
            x=bar_df["team"], y=bar_df["net_ppg"],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Net PPG: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Net Points Per Game",
                          showlegend=False,
                          xaxis_title="Team", yaxis_title="Net PPG")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Offense vs Defense scatter ───────────────────────────────────────────────
st.markdown("#### Offense vs Defense")

# Build the merged base DataFrame once, outside the fragment
abbr_col = "team_abbr" if "team_abbr" in teams.columns else "team"
div_col  = next((c for c in ["team_division", "division"] if c in teams.columns), None)
conf_col = next((c for c in ["team_conference", "conference"] if c in teams.columns), None)

merge_cols    = [abbr_col] + ([div_col] if div_col else []) + ([conf_col] if conf_col else [])
_scatter_base = full_df.merge(teams[merge_cols], left_on="team", right_on=abbr_col, how="left")

if conf_col is None and div_col is not None:
    _scatter_base["derived_conf"] = _scatter_base[div_col].apply(
        lambda x: str(x).split()[0] if pd.notna(x) and " " in str(x) else None
    )
    conf_col = "derived_conf"


# Pass data through session_state so the fragment takes no arguments.
# Fragment arguments that include DataFrames can cause Streamlit to fall back
# to a full page re-run instead of an isolated fragment re-run.
st.session_state["_sc_base"]    = _scatter_base
st.session_state["_sc_teams"]   = teams
st.session_state["_sc_div_col"] = div_col
st.session_state["_sc_conf_col"] = conf_col
st.session_state["_sc_tr_v"]    = _v  # pass reset version into fragment


@st.fragment
def _scatter_section():
    scatter_base = st.session_state["_sc_base"]
    _teams       = st.session_state["_sc_teams"]
    _div_col     = st.session_state["_sc_div_col"]
    _conf_col    = st.session_state["_sc_conf_col"]
    _fv          = st.session_state.get("_sc_tr_v", 0)  # version for reset

    # Filter row — two small columns above the chart
    fcol1, fcol2, _ = st.columns([1, 1, 3])

    with fcol1:
        if _conf_col:
            confs    = ["All"] + sorted(scatter_base[_conf_col].dropna().unique().tolist())
            sel_conf = st.selectbox("Conference", confs, key=f"s_conf_{_fv}")
        else:
            sel_conf = "All"

    with fcol2:
        if _div_col:
            div_src = (scatter_base if sel_conf == "All"
                       else scatter_base[scatter_base[_conf_col] == sel_conf])
            divs    = ["All"] + sorted(div_src[_div_col].dropna().unique().tolist())
            sel_div = st.selectbox("Division", divs, key=f"s_div_{_fv}")
        else:
            sel_div = "All"

    # Apply filters
    scatter_df = scatter_base.copy()
    if sel_conf != "All" and _conf_col:
        scatter_df = scatter_df[scatter_df[_conf_col] == sel_conf]
    if sel_div != "All" and _div_col:
        scatter_df = scatter_df[scatter_df[_div_col] == sel_div]

    for c in ["ppg", "oppg", "net_ppg"]:
        scatter_df[c] = scatter_df[c].round(1)

    if scatter_df.empty:
        st.info("No data for selected conference/division.")
        return

    avg_ppg  = scatter_df["ppg"].mean()
    avg_oppg = scatter_df["oppg"].mean()

    x_rng     = max(scatter_df["oppg"].max() - scatter_df["oppg"].min(), 1)
    y_rng     = max(scatter_df["ppg"].max()  - scatter_df["ppg"].min(),  1)
    logo_size = max(x_rng, y_rng) * 0.09

    fig2 = px.scatter(
        scatter_df, x="oppg", y="ppg",
        hover_name="team",
        custom_data=["team", "overall_rank", "ppg", "oppg", "net_ppg"],
    )
    fig2.update_traces(
        marker=dict(size=1, opacity=0),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Overall Rank: %{customdata[1]}<br>"
            "PPG: %{customdata[2]:.1f}<br>"
            "OPPG: %{customdata[3]:.1f}<br>"
            "Net PPG: %{customdata[4]:.1f}<extra></extra>"
        ),
    )

    for _, row in scatter_df.iterrows():
        url = get_logo(row["team"], _teams)
        if url:
            fig2.add_layout_image(dict(
                source=url, xref="x", yref="y",
                x=row["oppg"], y=row["ppg"],
                sizex=logo_size, sizey=logo_size,
                xanchor="center", yanchor="middle", layer="above",
            ))

    fig2.add_hline(y=avg_ppg,  line_dash="dot", line_color="rgba(79,70,229,0.25)")
    fig2.add_vline(x=avg_oppg, line_dash="dot", line_color="rgba(79,70,229,0.25)")

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title="Offense vs Defense",
        xaxis_title="Defensive PPG Allowed (lower = better →)",
        yaxis_title="Offensive PPG Scored",
    )
    fig2.update_xaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)


_scatter_section()

# ── About ────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **v1 — Simple scoring-based ratings**
    - **PPG** = points scored per game
    - **OPPG** = points allowed per game
    - **Net PPG** = PPG − OPPG

    Upcoming: opponent-adjusted ratings, EPA-based efficiency, win probability model.
    """)
