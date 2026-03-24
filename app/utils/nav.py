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

    # ── Mobile bottom nav (rendered via st.markdown — pure HTML, no JS) ───────
    st.markdown("""
    <div class="mobile-nav-bar">
      <a href="/"             class="mobile-nav-item"><span class="nav-emoji">🏠</span><span>Home</span></a>
      <a href="/Team_Ratings" class="mobile-nav-item"><span class="nav-emoji">📊</span><span>Ratings</span></a>
      <a href="/Player_Stats" class="mobile-nav-item"><span class="nav-emoji">🏃</span><span>Players</span></a>
      <a href="/Schedule"     class="mobile-nav-item"><span class="nav-emoji">📅</span><span>Schedule</span></a>
      <a href="/Historical"   class="mobile-nav-item"><span class="nav-emoji">📈</span><span>History</span></a>
      <a href="/Fantasy"      class="mobile-nav-item"><span class="nav-emoji">🏆</span><span>Fantasy</span></a>
    </div>
    """, unsafe_allow_html=True)

    # ── Mobile filter button + sidebar toggle ─────────────────────────────────
    # st.markdown strips ALL <script> and onclick handlers, so we CANNOT use it
    # for interactive elements.  Instead, components.html() runs in an iframe
    # where JS executes.  The script creates the filter button directly in the
    # PARENT document and attaches the click handler there.
    components.html("""
    <script>
    (function() {
        var pdoc = window.parent.document;

        // Guard: only create the button once, only on narrow screens
        if (pdoc.getElementById('nfl-filter-btn')) return;
        if (window.parent.innerWidth > 768) return;

        // ── Create the floating filter button in the parent DOM ──
        var btn = pdoc.createElement('div');
        btn.id = 'nfl-filter-btn';
        btn.textContent = '\\u2630 Filters';       // ☰ Filters
        btn.setAttribute('role', 'button');
        btn.setAttribute('tabindex', '0');
        btn.style.cssText = [
            'position:fixed',
            'right:14px',
            'bottom:108px',                         // above the raised nav bar
            'z-index:999998',
            'background:#6b7280',
            'color:#fff',
            'border:none',
            'border-radius:20px',
            'padding:8px 14px 8px 11px',
            'font-size:0.78rem',
            'font-weight:600',
            'cursor:pointer',
            'box-shadow:0 2px 10px rgba(0,0,0,0.22)',
            'display:flex',
            'align-items:center',
            'gap:5px',
            'font-family:Inter,sans-serif',
            '-webkit-tap-highlight-color:transparent',
            'letter-spacing:0.02em',
        ].join(';');

        // ── Toggle sidebar on tap ──
        btn.addEventListener('click', function() {
            var sb = pdoc.querySelector('[data-testid="stSidebar"]');
            if (!sb) return;
            sb.classList.toggle('sidebar-open');
        });

        // ── Optional: close sidebar when tapping outside it ──
        pdoc.addEventListener('click', function(e) {
            var sb = pdoc.querySelector('[data-testid="stSidebar"]');
            if (!sb || !sb.classList.contains('sidebar-open')) return;
            // If the click is inside the sidebar or on the filter button, ignore
            if (sb.contains(e.target) || e.target.id === 'nfl-filter-btn') return;
            sb.classList.remove('sidebar-open');
        });

        pdoc.body.appendChild(btn);
    })();
    </script>
    """, height=0)
