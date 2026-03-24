"""Shared sidebar navigation component for all pages."""
import streamlit as st


def render_sidebar_nav(current_page: str = ""):
    """Render the branded sidebar header followed by manual page links.

    We use st.page_link instead of Streamlit's auto-discovered nav so we
    control the order: brand → nav links → page-specific filters.
    """
    # ── Brand header ──────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <span class="logo-text">NFL <span class="accent">Analytics</span></span>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    # ── Page links ────────────────────────────────────────────────────────────
    st.sidebar.page_link("main.py",                    label="Home",             icon="🏠")
    st.sidebar.page_link("pages/1_Team_Ratings.py",    label="Team Ratings",     icon="📊")
    st.sidebar.page_link("pages/2_Player_Stats.py",    label="Player Stats",     icon="🏃")
    st.sidebar.page_link("pages/3_Schedule.py",        label="Schedule",         icon="📅")
    st.sidebar.page_link("pages/4_Historical.py",      label="Historical Trends",icon="📈")
    st.sidebar.page_link("pages/5_Fantasy.py",         label="Fantasy Football", icon="🏆")

    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
