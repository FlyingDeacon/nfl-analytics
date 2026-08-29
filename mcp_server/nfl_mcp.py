#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp>=2.0",
#     "duckdb>=1.0",
#     "pandas>=2.0,<3",
#     "numpy>=1.26",
#     "pyarrow>=15",
# ]
# ///
"""MCP server exposing the NFL analytics dashboard's data to Claude.

Run by `uv run`, which builds the dependency environment on demand - so this
stays independent of the dashboard's Python 3.9 venv.

pandas is pinned below 3.0 deliberately: the dashboard runs pandas 2.x, and
matching it keeps answers here identical to what the pages render.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The shared data layer lives in the dashboard's app/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from utils import nfl_data_core as core  # noqa: E402

_con = None


def connection():
    """Build the DuckDB view layer once, on first use."""
    global _con
    if _con is None:
        _con = core.build_connection()
    return _con


INSTRUCTIONS = """\
This server exposes Brandon's NFL analytics dashboard as queryable tables.

Use `nfl_schema` first to see the tables and columns, then `nfl_query` to
answer questions with SQL. Never state an NFL statistic from memory when these
tools can produce it - the numbers here are the dashboard's own, including its
2026 fantasy projections and season simulation.

Data spans 2016-2025 for actual results; 2026 figures are model projections.
"""

server = MCPServer(name="nfl-analytics", instructions=INSTRUCTIONS)


@server.tool(
    description=(
        "Show the available NFL tables, their columns and row counts, plus "
        "notes on what each column means. Call this before writing a query."
    )
)
def nfl_schema() -> str:
    return core.render_schema(connection())


@server.tool(
    description=(
        "Run a read-only DuckDB SELECT against the NFL datasets (player game "
        "logs 2016-2025, schedules, team ratings, 2026 fantasy big boards, and "
        "the 2026 season simulation). Returns CSV rows. Use nfl_schema first "
        "if you are unsure of the columns."
    )
)
def nfl_query(sql: str) -> str:
    """Args:
    sql: A single SELECT or WITH statement, without a trailing semicolon.
    """
    return core.run_sql(connection(), sql)


@server.tool(
    description=(
        "Resolve a partial, misspelled, or nickname player reference to the "
        "exact name string used in the data. Use before querying by name."
    )
)
def nfl_find_player(name: str) -> str:
    """Args:
    name: Full or partial player name, e.g. "mahomes" or "Ja'Marr".
    """
    return core.find_player(connection(), name)


if __name__ == "__main__":
    server.run(transport="stdio")
