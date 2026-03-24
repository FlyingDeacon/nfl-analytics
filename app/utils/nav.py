"""Shared sidebar navigation component for all pages."""
import streamlit as st
import streamlit.components.v1 as components


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

    # ── Mobile bottom nav (only visible on small screens via CSS) ─────────────
    # NOTE: st.markdown strips <script> and onclick, so the nav bar is pure
    # HTML links (no JS needed).  The filter button is a plain element whose
    # click handler is wired up by the components.html() block below.
    st.markdown("""
    <div class="mobile-nav-bar">
      <a href="/"             class="mobile-nav-item"><span class="nav-emoji">🏠</span><span>Home</span></a>
      <a href="/Team_Ratings" class="mobile-nav-item"><span class="nav-emoji">📊</span><span>Ratings</span></a>
      <a href="/Player_Stats" class="mobile-nav-item"><span class="nav-emoji">🏃</span><span>Players</span></a>
      <a href="/Schedule"     class="mobile-nav-item"><span class="nav-emoji">📅</span><span>Schedule</span></a>
      <a href="/Historical"   class="mobile-nav-item"><span class="nav-emoji">📈</span><span>History</span></a>
      <a href="/Fantasy"      class="mobile-nav-item"><span class="nav-emoji">🏆</span><span>Fantasy</span></a>
    </div>

    <!-- Filter button (visible on mobile only via CSS).
         The onclick is wired by the components.html JS below. -->
    <div class="mobile-filter-btn" id="mobile-filter-btn" role="button"
         aria-label="Toggle filters" tabindex="0">
      &#9776; Filters
    </div>
    """, unsafe_allow_html=True)

    # ── JS sidebar toggle ─────────────────────────────────────────────────────
    # components.html() renders in an iframe where <script> actually executes.
    # The script reaches into window.parent.document to:
    #   1. Find our #mobile-filter-btn in the parent DOM
    #   2. Attach a click handler that toggles the Streamlit sidebar
    components.html("""
    <script>
    (function() {
        function setup() {
            var doc = window.parent.document;

            // Wait for the button to appear in the parent DOM
            var btn = doc.getElementById('mobile-filter-btn');
            if (!btn) { setTimeout(setup, 300); return; }

            btn.onclick = function() {
                // 1. Try Streamlit's own toggle buttons (only if visible)
                var selectors = [
                    '[data-testid="stSidebarCollapsedControl"] button',
                    '[data-testid="stSidebarCollapseButton"]',
                    'button[aria-label="Open sidebar"]',
                    'button[aria-label="Close sidebar"]',
                ];
                for (var i = 0; i < selectors.length; i++) {
                    var toggle = doc.querySelector(selectors[i]);
                    if (toggle && toggle.offsetParent !== null) {
                        toggle.click();
                        return;
                    }
                }
                // 2. Fallback: toggle sidebar aria-expanded + transform
                var sb = doc.querySelector('[data-testid="stSidebar"]');
                if (sb) {
                    var isOpen = sb.getAttribute('aria-expanded') === 'true';
                    sb.setAttribute('aria-expanded', isOpen ? 'false' : 'true');
                    sb.style.transform = isOpen ? 'translateX(-100%)' : 'translateX(0)';
                    sb.style.transition = 'transform 0.3s ease';
                    if (!isOpen) sb.style.zIndex = '999997';
                }
            };
        }
        setup();
    })();
    </script>
    """, height=0)
