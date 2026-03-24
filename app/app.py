from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import textwrap

TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#00338D",
    "CAR": "#0085CA",
    "CHI": "#0B162A",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#003594",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#203731",
    "HOU": "#03202F",
    "IND": "#002C5F",
    "JAX": "#006778",
    "KC": "#E31837",
    "LV": "#000000",
    "LAC": "#0080C6",
    "LAR": "#003594",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#002244",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#002244",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#4B92DB",
    "WAS": "#5A1414"
}


def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    ratings_path = base_dir / "data" / "processed" / "team_ratings.csv"
    teams_path = base_dir / "data" / "raw" / "teams.csv"

    if not ratings_path.exists():
        st.error("team_ratings.csv not found. Run build_team_ratings.py first.")
        st.stop()

    if not teams_path.exists():
        st.error("teams.csv not found. Run load_nfl_data.py first.")
        st.stop()

    ratings = pd.read_csv(ratings_path)
    teams = pd.read_csv(teams_path)
    return ratings, teams


def render_team_card(title, team_name, value_text, logo_url, rank_label=None, rank_value=None):
    logo_html = ""
    if pd.notna(logo_url) and str(logo_url).strip() != "":
        logo_html = (
            '<div style="margin: 8px 0; text-align: center;">'
            f'<img src="{logo_url}" width="60">'
            '</div>'
        )

    rank_html = ""
    if rank_label is not None and rank_value is not None and pd.notna(rank_value):
        rank_html = (
            '<div style="color:#6b7280; margin-top:6px; font-size:0.95rem;">'
            f'{rank_label}: #{int(rank_value)}'
            '</div>'
        )

    card_html = (
        '<div style="text-align:center;">'
        f'<div style="font-weight:600; margin-bottom:8px;">{title}</div>'
        f'{logo_html}'
        f'<div style="font-size:1.2rem; font-weight:700;">{team_name}</div>'
        f'<div style="color:#6b7280; margin-top:4px;">{value_text}</div>'
        f'{rank_html}'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def build_logo_scatter(chart_df, get_logo):
    fig = px.scatter(
        chart_df,
        x="oppg",
        y="ppg",
        hover_name="team",
        custom_data=["team", "overall_rank", "ppg", "oppg", "net_ppg"]
    )

    fig.update_traces(
        marker=dict(size=1, opacity=0),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Overall Rank: %{customdata[1]}<br>"
            "PPG: %{customdata[2]:.1f}<br>"
            "OPPG: %{customdata[3]:.1f}<br>"
            "Net PPG: %{customdata[4]:.1f}<extra></extra>"
        )
    )

    # Dynamic logo sizing (so logos don't get huge when filtering)
    x_range = chart_df["oppg"].max() - chart_df["oppg"].min()
    y_range = chart_df["ppg"].max() - chart_df["ppg"].min()
    logo_size = max(x_range, y_range) * 0.08

    # Add team logos
    for _, row in chart_df.iterrows():
        logo_url = get_logo(row["team"])
        if pd.notna(logo_url) and str(logo_url).strip() != "":
            fig.add_layout_image(
                dict(
                    source=logo_url,
                    xref="x",
                    yref="y",
                    x=row["oppg"],
                    y=row["ppg"],
                    sizex=logo_size,
                    sizey=logo_size,
                    xanchor="center",
                    yanchor="middle",
                    layer="above"
                )
            )

    # --- ADD QUADRANT LINES ---
    avg_ppg = chart_df["ppg"].mean()
    avg_oppg = chart_df["oppg"].mean()

    fig.add_hline(
        y=avg_ppg,
        line_dash="dot",
        line_color="rgba(200,200,200,0.6)"
    )

    fig.add_vline(
        x=avg_oppg,
        line_dash="dot",
        line_color="rgba(200,200,200,0.6)"
    )
    # --- END ADDITION ---

    fig.update_layout(
        title="Offense vs Defense",
        xaxis_title="Defensive PPG Allowed (lower is better → right)",
        yaxis_title="Offensive PPG Scored",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Reverse x-axis (better defense → right side)
    fig.update_xaxes(autorange="reversed")

    return fig


def main():
    st.set_page_config(
        page_title="NFL Analytics Dashboard",
        page_icon="🏈",
        layout="wide"
    )

    ratings, teams = load_data()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
        .main-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            text-align: center;
            color: #1e1e2e;
        }
        .sub-text {
            color: #6b7280;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">🏈 NFL Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">Prototype dashboard for team ratings, rankings, and future predictive models.</div>',
        unsafe_allow_html=True
    )

    st.sidebar.title("Menu")

    seasons = sorted(ratings["season"].dropna().unique(), reverse=True)
    selected_season = st.sidebar.selectbox("Season", seasons)

    full_season_df = ratings[ratings["season"] == selected_season].copy()

    team_list = ["All Teams"] + sorted(full_season_df["team"].unique().tolist())
    selected_team = st.sidebar.selectbox("Team", team_list)

    season_df = full_season_df.copy()
    if selected_team != "All Teams":
        season_df = season_df[season_df["team"] == selected_team].copy()

    sort_metric = st.sidebar.selectbox(
        "Sort by",
        ["net_ppg", "ppg", "oppg", "games"]
    )

    ascending = sort_metric == "oppg"
    season_df = season_df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)

    overall_rank_df = full_season_df.sort_values("net_ppg", ascending=False).reset_index(drop=True)
    overall_rank_df["overall_rank"] = range(1, len(overall_rank_df) + 1)

    offense_rank_df = full_season_df.sort_values("ppg", ascending=False).reset_index(drop=True)
    offense_rank_df["offense_rank"] = range(1, len(offense_rank_df) + 1)

    defense_rank_df = full_season_df.sort_values("oppg", ascending=True).reset_index(drop=True)
    defense_rank_df["defense_rank"] = range(1, len(defense_rank_df) + 1)

    net_rank_df = full_season_df.sort_values("net_ppg", ascending=False).reset_index(drop=True)
    net_rank_df["net_rank"] = range(1, len(net_rank_df) + 1)

    season_df = season_df.merge(
        overall_rank_df[["team", "overall_rank"]],
        on="team",
        how="left"
    ).merge(
        offense_rank_df[["team", "offense_rank"]],
        on="team",
        how="left"
    ).merge(
        defense_rank_df[["team", "defense_rank"]],
        on="team",
        how="left"
    ).merge(
        net_rank_df[["team", "net_rank"]],
        on="team",
        how="left"
    )

    team_lookup = teams.copy()

    if "team_abbr" in team_lookup.columns:
        abbr_col = "team_abbr"
    elif "team" in team_lookup.columns:
        abbr_col = "team"
    else:
        abbr_col = team_lookup.columns[0]

    logo_col = None
    for col in ["team_logo_espn", "team_logo_wikipedia", "team_logo_squared", "team_logo"]:
        if col in team_lookup.columns:
            logo_col = col
            break

    def get_logo(team_abbr):
        if logo_col is None:
            return None
        match = team_lookup[team_lookup[abbr_col] == team_abbr]
        if match.empty:
            return None
        return match.iloc[0][logo_col]

    st.markdown(
        "<h3 style='text-align: center; margin-bottom: 20px;'>Snapshot</h3>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    if not season_df.empty:
        best_offense = season_df.sort_values("ppg", ascending=False).iloc[0]
        best_defense = season_df.sort_values("oppg", ascending=True).iloc[0]
        best_net = season_df.sort_values("net_ppg", ascending=False).iloc[0]

        with c1:
            render_team_card(
                "Best Offense",
                best_offense["team"],
                f"{best_offense['ppg']:.1f} PPG",
                get_logo(best_offense["team"]),
                "Offense Rank",
                best_offense["offense_rank"]
            )

        with c2:
            render_team_card(
                "Best Defense",
                best_defense["team"],
                f"{best_defense['oppg']:.1f} OPPG",
                get_logo(best_defense["team"]),
                "Defense Rank",
                best_defense["defense_rank"]
            )

        with c3:
            render_team_card(
                "Best Net Rating",
                best_net["team"],
                f"{best_net['net_ppg']:.1f} Net PPG",
                get_logo(best_net["team"]),
                "Net Rank",
                best_net["net_rank"]
            )
    else:
       
        with c1:
            st.markdown("<div style='text-align:center; font-weight:600;'>Best Offense</div><div style='text-align:center;'>N/A</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='text-align:center; font-weight:600;'>Best Defense</div><div style='text-align:center;'>N/A</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div style='text-align:center; font-weight:600;'>Best Net Rating</div><div style='text-align:center;'>N/A</div>", unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Team Ratings")

        display_df = season_df.copy()
        for col in ["ppg", "oppg", "net_ppg"]:
            display_df[col] = display_df[col].round(1)

        st.dataframe(
            display_df[["team", "overall_rank", "games", "ppg", "oppg", "net_ppg"]],
            width="stretch",
            hide_index=True
        )

    with right:
        st.markdown("### Net Rating")

        if not season_df.empty:
            chart_df = season_df.sort_values("net_ppg", ascending=False)

            fig = px.bar(
                chart_df,
                x="team",
                y="net_ppg",
                title="Net Points Per Game",
                color="team",
                color_discrete_map=TEAM_COLORS
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Team",
                yaxis_title="Net PPG",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available for the selected filters.")

    st.markdown("---")

    st.markdown("### Offense vs Defense Scatter")

    if not full_season_df.empty:
        scatter_df = full_season_df.merge(
            overall_rank_df[["team", "overall_rank"]],
            on="team",
            how="left"
        ).copy()

        if "team_abbr" in teams.columns:
            scatter_abbr_col = "team_abbr"
        else:
            scatter_abbr_col = "team"

        scatter_division_col = None
        for col in ["team_division", "division"]:
            if col in teams.columns:
                scatter_division_col = col
                break

        scatter_conference_col = None
        for col in ["team_conference", "conference"]:
            if col in teams.columns:
                scatter_conference_col = col
                break

        merge_cols = [scatter_abbr_col]
        if scatter_division_col is not None:
            merge_cols.append(scatter_division_col)
        if scatter_conference_col is not None:
            merge_cols.append(scatter_conference_col)

        scatter_df = scatter_df.merge(
            teams[merge_cols],
            left_on="team",
            right_on=scatter_abbr_col,
            how="left"
        )

        # If conference column doesn't exist, derive it from division
        if scatter_conference_col is None and scatter_division_col is not None:
            scatter_conference_col = "derived_conference"
            scatter_df[scatter_conference_col] = scatter_df[scatter_division_col].apply(
                lambda x: str(x).split()[0] if pd.notna(x) and " " in str(x) else None
            )

        filter_col1, filter_col2, _ = st.columns([1, 1, 3])

        with filter_col1:
            if scatter_conference_col is not None:
                conferences = ["All"] + sorted(
                    scatter_df[scatter_conference_col].dropna().unique().tolist()
                )
                selected_conference = st.selectbox(
                    "Conference",
                    conferences,
                    key="scatter_conf"
                )
            else:
                selected_conference = "All"

        with filter_col2:
            if scatter_division_col is not None:
                division_source_df = scatter_df.copy()

                if selected_conference != "All" and scatter_conference_col is not None:
                    division_source_df = division_source_df[
                        division_source_df[scatter_conference_col] == selected_conference
                    ]

                divisions = ["All"] + sorted(
                    division_source_df[scatter_division_col].dropna().unique().tolist()
                )

                selected_division = st.selectbox(
                    "Division",
                    divisions,
                    key="scatter_division"
                )
            else:
                selected_division = "All"

        if selected_conference != "All" and scatter_conference_col is not None:
            scatter_df = scatter_df[
                scatter_df[scatter_conference_col] == selected_conference
            ].copy()

        if selected_division != "All" and scatter_division_col is not None:
            scatter_df = scatter_df[
                scatter_df[scatter_division_col] == selected_division
            ].copy()

        scatter_df["ppg"] = scatter_df["ppg"].round(1)
        scatter_df["oppg"] = scatter_df["oppg"].round(1)
        scatter_df["net_ppg"] = scatter_df["net_ppg"].round(1)

        scatter_fig = build_logo_scatter(scatter_df, get_logo)
        st.plotly_chart(scatter_fig, width="stretch")
    else:
        st.info("No data available for scatter plot.")
    st.markdown("---")

    with st.expander("About this model"):
        st.write(
            """
            This first version uses simple scoring-based ratings:
            - **PPG** = points scored per game
            - **OPPG** = points allowed per game
            - **Net PPG** = PPG minus OPPG

            Later versions can add opponent adjustments, efficiency, EPA-based modeling,
            and matchup predictions.
            """
        )


if __name__ == "__main__":
    main()