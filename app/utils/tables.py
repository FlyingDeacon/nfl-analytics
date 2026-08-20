"""Shared helpers for rendering st.dataframe columns.

Kept separate from utils/styles.py, which holds only CSS/colour constants and
deliberately has no pandas dependency.
"""
from __future__ import annotations

import pandas as pd

# Marker shown on a board for players a source never ranked.
UNRANKED_MARK = "\u2014"


def rank_display(s: pd.Series) -> pd.Series:
    """Format a rank column so clicking its header sorts the way it reads.

    Streamlit sorts a *numeric* column's blanks to the top on ascending, which
    buries rank 1 under every unranked player. Text columns sort by collation
    instead, so render the ranks right-aligned to a fixed width: the padding
    keeps 10 after 9 (plain text would give 1, 10, 2), and the one extra space
    guarantees every number leads with whitespace, which the collator orders
    ahead of the em dash. Unranked players therefore land at the bottom.
    """
    nums = pd.to_numeric(s, errors="coerce")
    width = (len(str(int(nums.max()))) + 1) if nums.notna().any() else 2
    return nums.map(lambda v: UNRANKED_MARK if pd.isna(v) else f"{int(v):{width}d}")
