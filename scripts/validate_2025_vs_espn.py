"""One-off validation: how would the current model have done on 2025?

Runs the projection engine walk-forward as of 2025 (trains only on <2025, projects
2025 off 2024 features), then compares it against actual 2025 fantasy finishes for
the preseason consensus top-50 board (data/raw/preseason_rankings.csv). The board's
own preseason rank is scored against the same actuals, so we can see whether the
model would have beaten the market on those 50 players.

    python scripts/validate_2025_vs_espn.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from scipy.stats import spearmanr

from model.projection import ProjectionConfig, build_predictions_core

SEASON = 2025
TOP_N = 50


def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = re.sub(r"\bo\b$", "", name).strip()   # strip stray trailing "O" ADP artifact
    name = re.sub(r"[.\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def main() -> None:
    weekly = pd.read_csv(REPO_ROOT / "data" / "raw" / "weekly.csv", low_memory=False)
    weekly.columns = [c.lower().strip() for c in weekly.columns]

    name_col = "player_display_name"
    pos_col = "position"
    track_col = "player_id"
    config = ProjectionConfig(
        target_col="fantasy_points_ppr", name_col=name_col, pos_col=pos_col,
        team_col="recent_team", track_col=track_col,
    )

    # ── Model projections for 2025 (walk-forward, no leakage) ────────────────
    proj, _ = build_predictions_core(weekly, config, as_of_season=SEASON)
    proj = proj[[name_col, pos_col, "predicted_pts"]].copy()
    proj["proj_rank"] = proj["predicted_pts"].rank(ascending=False, method="min").astype(int)
    proj["key"] = proj[name_col].map(_norm)

    # ── Actual 2025 finishes (overall PPR total) ─────────────────────────────
    reg = weekly[(weekly["season"] == SEASON) & (weekly.get("season_type", "REG") == "REG")]
    actual = reg.groupby([name_col, pos_col], as_index=False)["fantasy_points_ppr"].sum()
    actual.rename(columns={"fantasy_points_ppr": "actual_pts"}, inplace=True)
    actual["actual_rank"] = actual["actual_pts"].rank(ascending=False, method="min").astype(int)
    actual["key"] = actual[name_col].map(_norm)

    # ── Preseason consensus top-50 board ─────────────────────────────────────
    pre = pd.read_csv(REPO_ROOT / "data" / "raw" / "preseason_rankings.csv")
    pre = pre[pre["season"] == SEASON].nsmallest(TOP_N, "preseason_rank").copy()
    pre["key"] = pre["player_name"].map(_norm)

    board = pre.merge(actual[["key", "actual_pts", "actual_rank"]], on="key", how="left")
    board = board.merge(proj[["key", "predicted_pts", "proj_rank"]], on="key", how="left")

    # Rank the two forecasts WITHIN the top-50 set so they're compared on equal footing
    board = board.sort_values("preseason_rank").reset_index(drop=True)
    matched = board.dropna(subset=["actual_rank"]).copy()
    matched["actual_rank_within"] = matched["actual_pts"].rank(ascending=False, method="min")
    matched["pre_rank_within"] = matched["preseason_rank"].rank(ascending=True, method="min")
    have_proj = matched.dropna(subset=["proj_rank"]).copy()
    have_proj["proj_rank_within"] = have_proj["predicted_pts"].rank(ascending=False, method="min")

    # ── Per-player table ─────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"2025 VALIDATION — preseason top-{TOP_N} board vs MODEL vs ACTUAL")
    print(f"{'='*92}")
    print(f"{'Pre':>3} {'Player':<24} {'Pos':<3} {'Model':>6} {'Actual':>6} "
          f"{'PreErr':>7} {'ModErr':>7}")
    for _, r in board.iterrows():
        mr = "" if pd.isna(r["proj_rank"]) else f"{int(r['proj_rank'])}"
        ar = "" if pd.isna(r["actual_rank"]) else f"{int(r['actual_rank'])}"
        pre_err = "" if pd.isna(r["actual_rank"]) else f"{int(r['preseason_rank']-r['actual_rank']):+d}"
        mod_err = "" if (pd.isna(r["actual_rank"]) or pd.isna(r["proj_rank"])) \
            else f"{int(r['proj_rank']-r['actual_rank']):+d}"
        print(f"{int(r['preseason_rank']):>3} {r['player_name'][:24]:<24} "
              f"{str(r.get('position') or '')[:3]:<3} {mr:>6} {ar:>6} {pre_err:>7} {mod_err:>7}")

    unmatched = board[board["actual_rank"].isna()]["player_name"].tolist()
    if unmatched:
        print(f"\n(No 2025 actuals matched for: {', '.join(unmatched)})")

    # ── Aggregate accuracy ───────────────────────────────────────────────────
    pre_rho, _ = spearmanr(matched["pre_rank_within"], matched["actual_rank_within"])
    pre_mae = (matched["pre_rank_within"] - matched["actual_rank_within"]).abs().mean()
    mod_rho, _ = spearmanr(have_proj["proj_rank_within"], have_proj["actual_pts"].rank(ascending=False, method="min"))
    # recompute actual-within on the have_proj subset for a fair pairing
    have_proj["actual_rank_within2"] = have_proj["actual_pts"].rank(ascending=False, method="min")
    mod_rho, _ = spearmanr(have_proj["proj_rank_within"], have_proj["actual_rank_within2"])
    mod_mae = (have_proj["proj_rank_within"] - have_proj["actual_rank_within2"]).abs().mean()

    # Did they hit the overall top-50? (using full-pool actual_rank)
    pre_hit = (matched["actual_rank"] <= TOP_N).mean()
    mod_hit = (have_proj["actual_rank"] <= TOP_N).mean()

    print(f"\n{'='*92}")
    print(f"ACCURACY on the preseason top-{TOP_N} (ranks scored within this set):")
    print(f"  matched to 2025 actuals:  board={len(matched)}   model={len(have_proj)}")
    print(f"  Spearman rank corr:       board={pre_rho:.3f}    model={mod_rho:.3f}")
    print(f"  Mean abs rank error:      board={pre_mae:.2f}      model={mod_mae:.2f}")
    print(f"  % who finished overall top-{TOP_N}: board={pre_hit:.0%}   model-picks={mod_hit:.0%}")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
