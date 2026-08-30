"""Passcode gate for the private Season Projections page.

The rest of the dashboard is meant to be public, but the CHOPPED survivor
planner is an edge in a pool with real money and real entrants — publishing the
optimal week-by-week plan would hand it to everyone we are playing against.

This is deliberately a shared secret, not a login: one passcode, no accounts,
and it guards a pool strategy rather than anything sensitive. It runs
server-side and st.stop() halts the script before the page renders, so the
protected content is never sent to the browser.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

SESSION_KEY = "projections_unlocked"


def _expected_passcode() -> str:
    """Configured passcode, or "" when none is set anywhere."""
    try:
        code = st.secrets.get("app", {}).get("projections_passcode", "")
    except Exception:          # no secrets.toml at all (fresh clone, some CI)
        code = ""
    return str(code or os.environ.get("PROJECTIONS_PASSCODE", ""))


def is_unlocked() -> bool:
    """True once this browser session has entered the passcode."""
    return bool(st.session_state.get(SESSION_KEY))


def require_passcode(title: str = "Private page") -> None:
    """Show a passcode prompt and halt the script unless it has been entered.

    Fails closed when no passcode is configured — a deploy that forgot to set
    the secret should hide the page, not publish it.
    """
    if is_unlocked():
        return

    expected = _expected_passcode()

    st.markdown(f"""
    <div class="nfl-page-header">
        <div class="icon">🔒</div>
        <div>
            <div class="title">{title}</div>
            <div class="subtitle">Enter the passcode to continue</div>
        </div>
    </div>
    <div class="gold-rule"></div>
    """, unsafe_allow_html=True)

    if not expected:
        st.error("No passcode is configured, so this page is locked. Set "
                 "`[app] projections_passcode` in `.streamlit/secrets.toml` "
                 "(or the `PROJECTIONS_PASSCODE` environment variable).")
        st.stop()

    with st.form("passcode_gate"):
        entered = st.text_input("Passcode", type="password",
                                label_visibility="collapsed",
                                placeholder="Passcode")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        # compare_digest rather than == so a wrong guess takes the same time
        # regardless of how many leading characters it got right.
        if hmac.compare_digest(entered, expected):
            st.session_state[SESSION_KEY] = True
            st.rerun()
        st.error("Incorrect passcode.")

    st.stop()
