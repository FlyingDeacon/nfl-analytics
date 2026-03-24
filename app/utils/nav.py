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

    # ── Mobile bottom nav + filter button (only visible on small screens via CSS) ──
    st.markdown("""
    <div class="mobile-nav-bar">
      <a href="/"             class="mobile-nav-item"><span class="nav-emoji">🏠</span><span>Home</span></a>
      <a href="/Team_Ratings" class="mobile-nav-item"><span class="nav-emoji">📊</span><span>Ratings</span></a>
      <a href="/Player_Stats" class="mobile-nav-item"><span class="nav-emoji">🏃</span><span>Players</span></a>
      <a href="/Schedule"     class="mobile-nav-item"><span class="nav-emoji">📅</span><span>Schedule</span></a>
      <a href="/Historical"   class="mobile-nav-item"><span class="nav-emoji">📈</span><span>History</span></a>
      <a href="/Fantasy"      class="mobile-nav-item"><span class="nav-emoji">🏆</span><span>Fantasy</span></a>
    </div>

    <!-- Grey filter arrow button — toggles the sidebar on mobile -->
    <button class="mobile-filter-btn" onclick="toggleSidebar()" aria-label="Toggle filters">
      &#9776; Filters
    </button>

    <script>
    function toggleSidebar() {
        // Try every known Streamlit sidebar toggle selector
        var selectors = [
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapseButton"]',
            'button[aria-label="Open sidebar"]',
            'button[aria-label="Close sidebar"]',
            'button[aria-controls="bui3-tabpanel-0"]',
            '[data-testid="collapsedControl"] button',
            'section[data-testid="stSidebar"] button',
        ];
        for (var i = 0; i < selectors.length; i++) {
            var btn = document.querySelector(selectors[i]);
            if (btn) { btn.click(); return; }
        }
        // Fallback: toggle sidebar visibility directly
        var sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            sb.style.transform = sb.style.transform === 'none' ? 'translateX(-110%)' : 'none';
        }
    }
    </script>
    """, unsafe_allow_html=True)
