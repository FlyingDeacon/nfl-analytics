TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB":  "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC":  "#E31837",
    "LV":  "#A5ACAF", "LAC": "#0080C6", "LAR": "#FFA300", "LA":  "#FFA300", "MIA": "#008E97",
    "MIN": "#4F2683", "NE":  "#002244", "NO":  "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF":  "#AA0000", "TB":  "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414",
}

NFL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Hide ALL Streamlit chrome (toolbar, header, footer, deploy, profile) ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="stAppHeader"],
[data-testid="stMetaDatas"],
[data-testid="stDeployLogsButton"],
footer,
#MainMenu,
.stAppHeader,
.stDeployButton,
header[data-testid],
[data-testid="stBottom"] > div:empty { display: none !important; }

:root {
    --bg:         #f5f6fa;
    --surface:    #ffffff;
    --surface2:   #f0f1f6;
    --border:     #e2e5ef;
    --accent:     #4f46e5;
    --accent-soft:#eef2ff;
    --accent2:    #10b981;
    --accent2-soft:#ecfdf5;
    --text:       #1e1e2e;
    --text-sec:   #4a4e69;
    --muted:      #8b8fa8;
    --radius:     14px;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
    --shadow-lg:  0 12px 40px rgba(0,0,0,0.08);
    --glass:      rgba(255,255,255,0.72);
    --glass-border: rgba(255,255,255,0.6);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

[data-testid="stMainBlockContainer"] {
    padding: 1.5rem clamp(0.5rem, 2vw, 2rem) !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

/* ── Typography ── */
h1,h2,h3,h4 {
    font-family: 'DM Sans', 'Inter', sans-serif !important;
    letter-spacing: -0.01em;
    color: var(--text) !important;
    font-weight: 700 !important;
}

/* ── Form elements ── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm) !important;
    background: var(--surface) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
}

hr {
    border-color: var(--border) !important;
    margin: 1.8rem 0 !important;
    opacity: 0.6;
}

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #c8cdd8; border-radius:3px; }

/* ── Page header (glass card) ── */
.nfl-page-header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 1.2rem; padding: 1.2rem clamp(0.8rem, 2vw, 1.5rem);
    overflow: hidden; word-break: break-word;
    background: var(--glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    box-shadow: var(--shadow-md);
}
.nfl-page-header .icon {
    font-size: 2rem; line-height: 1;
    background: var(--accent-soft);
    width: 52px; height: 52px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 14px;
}
.nfl-page-header .title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.7rem; font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--text); line-height: 1;
}
.nfl-page-header .subtitle {
    font-size: 0.85rem; color: var(--muted);
    margin-top: 4px; letter-spacing: 0.01em;
}

/* ── Stat card (glass) ── */
.stat-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
.stat-card .label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted);
    font-weight: 600; margin-bottom: 8px;
}
.stat-card .value {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: var(--text); line-height: 1;
}
.stat-card .sub {
    font-size: 0.8rem; color: var(--text-sec);
    margin-top: 6px; font-weight: 500;
}

/* ── Accent rule (gradient) ── */
.accent-rule {
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), transparent);
    border: none;
    border-radius: 2px;
    margin: 0.4rem 0 1.5rem 0;
}
/* Keep backward compat with old class name */
.gold-rule {
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), transparent);
    border: none;
    border-radius: 2px;
    margin: 0.4rem 0 1.5rem 0;
}

/* ── Sidebar navigation ── */
[data-testid="stSidebar"] [data-testid="stPageLink"],
[data-testid="stSidebar"] .stPageLink {
    border-radius: 10px !important;
    transition: background 0.2s, border-color 0.2s;
}

.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0.6rem 0 1.2rem 0;
}
.sidebar-brand .logo-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.35rem; font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--text); line-height: 1;
}
.sidebar-brand .logo-text .accent { color: var(--accent); }
.sidebar-brand .logo-icon { font-size: 1.6rem; line-height: 1; }

.sidebar-divider {
    height: 1px; background: var(--border);
    margin: 0.8rem 0; border: none;
}

.sidebar-section-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: var(--muted);
    font-weight: 600;
    padding: 0.4rem 0 0.5rem 4px; margin: 0;
}

/* Hide Streamlit's auto-generated nav — we use manual st.page_link instead */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Style the manual page_link nav items */
[data-testid="stSidebar"] [data-testid="stPageLink"],
[data-testid="stSidebar"] .stPageLink {
    border-radius: 10px !important;
    transition: background 0.2s, border-color 0.2s;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text-sec) !important;
    padding: 0.45rem 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"][aria-current="page"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* Hide the Streamlit sidebar collapse label / keyboard shortcut text */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    font-size: 0 !important;
    visibility: hidden !important;
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    overflow: hidden !important;
}
/* Also target any stray tooltip/title text in the sidebar header area */
[data-testid="stSidebar"] > div:first-child > div:first-child > span,
[data-testid="stSidebar"] [title*="key"],
[data-testid="stSidebar"] [aria-label*="key"] {
    font-size: 0 !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}

/* Sidebar filter section */
.sidebar-filters-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: var(--muted);
    font-weight: 600;
    padding: 0.6rem 0 0.3rem 4px; margin: 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ── Responsive layout ── */

/* Columns wrap when narrow */
[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: min(320px, 100%) !important;
    flex: 1 1 320px !important;
}

/* ── Mobile bottom navigation bar (hidden on desktop, flex on mobile) ── */
.mobile-nav-bar {
    display: none;
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #ffffff;
    border-top: 1px solid var(--border);
    padding: 8px 0 calc(env(safe-area-inset-bottom, 8px) + 4px);
    z-index: 999999;
    box-shadow: 0 -4px 16px rgba(0,0,0,0.07);
    justify-content: space-around;
    align-items: center;
}
.mobile-nav-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    color: var(--muted);
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    gap: 3px;
    padding: 4px 6px;
    border-radius: 8px;
    transition: color 0.2s;
    min-width: 48px;
    -webkit-tap-highlight-color: transparent;
}
.mobile-nav-item:hover,
.mobile-nav-item.active { color: var(--accent) !important; }
.mobile-nav-item .nav-emoji { font-size: 1.25rem; line-height: 1; }

/* ── Mobile filter toggle button (hidden on desktop) ── */
.mobile-filter-btn {
    display: none;
    position: fixed;
    right: 14px;
    bottom: calc(68px + env(safe-area-inset-bottom, 8px));
    z-index: 999998;
    background: #6b7280;
    color: #fff;
    border: none;
    border-radius: 20px;
    padding: 8px 14px 8px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    align-items: center;
    gap: 5px;
    font-family: 'Inter', sans-serif;
    -webkit-tap-highlight-color: transparent;
    letter-spacing: 0.02em;
}

/* ── Mobile overrides ── */
@media (max-width: 768px) {
    .mobile-nav-bar   { display: flex !important; }
    .mobile-filter-btn { display: flex !important; }

    /* Extra bottom padding so content doesn't hide behind the nav bar */
    [data-testid="stMainBlockContainer"] {
        padding-bottom: 100px !important;
    }

    /* When sidebar opens on mobile, make it a high-z overlay */
    [data-testid="stSidebar"][aria-expanded="true"] {
        z-index: 999997 !important;
    }
}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, DM Sans, sans-serif", color="#1e1e2e", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(gridcolor="#e8eaef", linecolor="#e2e5ef",
               zerolinecolor="#e2e5ef", tickfont=dict(size=11, color="#4a4e69")),
    yaxis=dict(gridcolor="#e8eaef", linecolor="#e2e5ef",
               zerolinecolor="#e2e5ef", tickfont=dict(size=11, color="#4a4e69")),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e5ef",
                    font=dict(family="Inter, sans-serif", color="#1e1e2e")),
    title_font=dict(family="DM Sans, Inter, sans-serif", size=17, color="#1e1e2e"),
)
