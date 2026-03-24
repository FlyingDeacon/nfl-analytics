import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils.styles import NFL_CSS
from utils.nav import render_sidebar_nav

st.set_page_config(
    page_title="NFL Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(NFL_CSS, unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
render_sidebar_nav(current_page="Home")

st.markdown("""
<style>
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }

.home-wrap { max-width: 100%; margin: 0 auto; padding: 2rem 1rem 3rem; box-sizing: border-box; overflow: hidden; }

/* Hero */
.hero { text-align: center; padding: 2.5rem 0 2rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase;
    font-weight: 600;
    color: #4f46e5; background: #eef2ff;
    border: 1px solid #e0e7ff;
    padding: 6px 14px; border-radius: 24px;
    margin-bottom: 1.4rem;
}
.hero-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10b981;
    animation: float 3s ease-in-out infinite;
}
.hero-eyebrow {
    font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    font-weight: 600; color: #8b8fa8; margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(1.8rem, 5vw, 3.2rem); font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.05; color: #1e1e2e;
}
.hero-title .accent { color: #4f46e5; }
.hero-sub {
    color: #4a4e69; font-size: 1rem;
    margin: 1.2rem auto 0; line-height: 1.7;
    max-width: min(480px, 100%); font-weight: 400;
}
.hero-sub code {
    color: #4f46e5; background: #eef2ff;
    padding: 2px 8px; border-radius: 6px;
    font-size: 0.88rem; font-weight: 500;
}

/* Gradient divider */
.gradient-divider {
    width: 100px; height: 3px; margin: 1.6rem auto;
    background: linear-gradient(90deg, #4f46e5, #10b981, transparent);
    border: none; border-radius: 2px;
}

/* Stats strip */
.stats-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin: 2rem 0;
}
.strip-item {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}
.strip-item:hover { transform: translateY(-2px); }
.strip-num {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #4f46e5; line-height: 1;
}
.strip-lbl {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: #8b8fa8;
    font-weight: 600; margin-top: 6px;
}

/* Section label */
.section-lbl {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: #8b8fa8;
    font-weight: 600;
    margin-bottom: 1rem; padding-left: 2px;
}

/* Nav card links */
.nav-card-link {
    text-decoration: none; color: inherit; display: block;
}
.nav-card-link:hover { text-decoration: none; color: inherit; }

/* Nav cards */
.nav-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px;
    margin-bottom: 2.5rem;
}
.nav-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.65);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    display: flex; align-items: flex-start; gap: 1rem;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.nav-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.08);
}
.nav-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--nc-accent);
    border-radius: 0 2px 2px 0;
}
.nav-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: var(--nc-icon-bg);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.nav-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    color: #1e1e2e; line-height: 1;
}
.nav-desc {
    font-size: 0.82rem; color: #6b7280;
    margin-top: 6px; line-height: 1.55;
}
.nav-tag {
    display: inline-block; font-size: 0.67rem;
    text-transform: uppercase; letter-spacing: 0.07em;
    font-weight: 600;
    padding: 3px 8px; border-radius: 6px; margin-top: 10px;
    background: var(--nc-tag-bg, #eef2ff);
    color: var(--nc-tag-color, #4f46e5);
}

/* Footer */
.home-footer {
    border-top: 1px solid #e2e5ef; padding-top: 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
}
.footer-brand {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; font-weight: 700;
    color: #8b8fa8;
}
.footer-brand .accent { color: #4f46e5; }
.footer-note { font-size: 0.75rem; color: #b0b4c4; }
</style>

<div class="home-wrap">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-badge"><span class="dot"></span>2016 – 2026 Season Data</div>
    <div class="hero-eyebrow">NFL Analytics Dashboard</div>
    <div class="hero-title">Your Edge on<br><span class="accent">Every</span> Game</div>
    <div class="gradient-divider"></div>
    <div class="hero-sub">
        Ratings, player stats, schedules, and historical trends —
        powered by <code>nfl_data_py</code>
        across a decade of NFL data.
    </div>
  </div>

  <!-- Stats strip -->
  <div class="stats-strip">
    <div class="strip-item">
      <div class="strip-num">32</div>
      <div class="strip-lbl">Teams Tracked</div>
    </div>
    <div class="strip-item">
      <div class="strip-num">11</div>
      <div class="strip-lbl">Seasons of Data</div>
    </div>
    <div class="strip-item">
      <div class="strip-num">5</div>
      <div class="strip-lbl">Analytics Modules</div>
    </div>
  </div>

  <!-- Nav cards -->
  <div class="section-lbl">Navigate</div>
  <div class="nav-grid">
    <a href="/Team_Ratings" class="nav-card-link" style="--nc-accent:#4f46e5; --nc-icon-bg:#eef2ff; --nc-tag-bg:#eef2ff; --nc-tag-color:#4f46e5;">
      <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div>
          <div class="nav-title">Team Ratings</div>
          <div class="nav-desc">PPG, OPPG, net rating, and an offense vs. defense scatter with team logos.</div>
          <span class="nav-tag">Rankings · Scatter</span>
        </div>
      </div>
    </a>
    <a href="/Player_Stats" class="nav-card-link" style="--nc-accent:#3b82f6; --nc-icon-bg:#eff6ff; --nc-tag-bg:#eff6ff; --nc-tag-color:#3b82f6;">
      <div class="nav-card">
        <div class="nav-icon">🏃</div>
        <div>
          <div class="nav-title">Player Stats</div>
          <div class="nav-desc">Season leaders in passing, rushing, and receiving with full leaderboards.</div>
          <span class="nav-tag">Passing · Rushing · Receiving</span>
        </div>
      </div>
    </a>
    <a href="/Schedule" class="nav-card-link" style="--nc-accent:#10b981; --nc-icon-bg:#ecfdf5; --nc-tag-bg:#ecfdf5; --nc-tag-color:#059669;">
      <div class="nav-card">
        <div class="nav-icon">📅</div>
        <div>
          <div class="nav-title">Schedule</div>
          <div class="nav-desc">Full game results with scores and upcoming matchup listings by week.</div>
          <span class="nav-tag">Results · Upcoming</span>
        </div>
      </div>
    </a>
    <a href="/Historical" class="nav-card-link" style="--nc-accent:#f59e0b; --nc-icon-bg:#fffbeb; --nc-tag-bg:#fffbeb; --nc-tag-color:#d97706;">
      <div class="nav-card">
        <div class="nav-icon">📈</div>
        <div>
          <div class="nav-title">Historical</div>
          <div class="nav-desc">Multi-season trend lines and a league-wide performance heatmap.</div>
          <span class="nav-tag">Trends · Heatmap</span>
        </div>
      </div>
    </a>
    <a href="/Fantasy" class="nav-card-link" style="--nc-accent:#8b5cf6; --nc-icon-bg:#f5f3ff; --nc-tag-bg:#f5f3ff; --nc-tag-color:#7c3aed;">
      <div class="nav-card">
        <div class="nav-icon">🏆</div>
        <div>
          <div class="nav-title">Fantasy Football</div>
          <div class="nav-desc">Season totals, per-game averages, and weekly consistency for PPR leagues.</div>
          <span class="nav-tag">PPR · Leaders · Weekly</span>
        </div>
      </div>
    </a>
  </div>

  <!-- Footer -->
  <div class="home-footer">
    <div class="footer-brand">NFL <span class="accent">Analytics</span></div>
    <div class="footer-note">Data via nfl_data_py &nbsp;·&nbsp; Built with Streamlit</div>
  </div>

</div>
""", unsafe_allow_html=True)
