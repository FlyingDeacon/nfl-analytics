"""Walk-forward backtest for the fantasy projection engine in model/projection.py.

For each holdout season S, trains only on season pairs < S (via
build_predictions_core(..., as_of_season=S)) and projects season S off S-1
features, then compares the projection to what actually happened in season S.

Two VOR numbers are computed per player, deliberately from different sources:
  - "proj_vor"   = predicted_pts - REPLACEMENT_LEVEL_PPR[pos], the static
    2024-25-calibrated constant. NOTE: this no longer mirrors _assign_vor() in
    the live page, which now derives its baseline from the projection pool via
    derive_replacement_baseline(). This run is therefore the "before" baseline —
    still useful as the thing to measure against, but no longer a description of
    what the app does. Switching this over is the remaining half of Phase 2.
  - "actual_vor" = actual_pts - dynamic_baseline[pos], where dynamic_baseline
    is derived from season S's own real finishes via
    model.projection.derive_replacement_baseline() (the audit's Phase 2
    roster-slot algorithm). A hardcoded constant from a *different* season
    would be a hardcoded-baseline comparison of exactly the kind the audit
    calls out — so actual outcomes get a fair, self-derived baseline instead.

Usage:
    python scripts/backtest_model.py --season 2025
    python scripts/backtest_model.py --seasons 2021 2022 2023 2024 2025 --output baseline_metrics.json
    python scripts/backtest_model.py --seasons 2021 2022 2023 2024 2025 \
        --compare configA.json configB.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.projection import ProjectionConfig, build_predictions_core, derive_replacement_baseline

# ── League settings (mirrors the live 10-team PPR app; not yet wired into the
# app itself — that's Phase 2. Kept local to the backtest until then). ────────
LEAGUE_SIZE = 10
ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

# Static replacement levels, formerly matching the live app's
# SCORING_REPLACEMENT_LEVELS["PPR"] (app/pages/7_Fantasy_Predictions.py).
# The app has since moved to derive_replacement_baseline(); this dict is kept as
# the "before" reference so the change stays measurable. Retiring it here too is
# the remaining half of Phase 2, after which nothing needs keeping in sync.
REPLACEMENT_LEVEL_PPR = {"QB": 290, "RB": 185, "WR": 185, "TE": 170}

TOP_N_HIT_RATES = (24, 50)


def load_weekly(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data" / "raw" / "weekly.csv"
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def resolve_columns(weekly: pd.DataFrame) -> tuple[str, str, str | None, str]:
    name_col = next((c for c in ["player_display_name", "player_name", "name"] if c in weekly.columns), None)
    id_col   = next((c for c in ["player_id", "gsis_id"] if c in weekly.columns), None)
    team_col = next((c for c in ["recent_team", "posteam", "team"] if c in weekly.columns), None)
    pos_col  = next((c for c in ["position", "pos"] if c in weekly.columns), None)
    if not name_col or not pos_col:
        raise SystemExit("Required columns (player name, position) not found in weekly.csv.")
    track_col = id_col if id_col else name_col
    return name_col, pos_col, team_col, track_col


def actual_outcomes_for_season(weekly: pd.DataFrame, season: int, config: ProjectionConfig) -> pd.DataFrame:
    """Actual season totals + games played for `season`, one row per player/position."""
    reg = weekly.copy()
    if "season_type" in reg.columns:
        reg = reg[reg["season_type"] == "REG"]
    reg = reg[(reg["season"] == season) & (reg[config.pos_col].isin(config.position_features))]

    group_keys = [config.track_col, config.name_col, config.pos_col]
    agg = reg.groupby(group_keys, as_index=False)[config.target_col].sum()
    gp  = reg.groupby(group_keys, as_index=False)[config.target_col].count()
    gp.rename(columns={config.target_col: "games"}, inplace=True)
    agg = agg.merge(gp, on=group_keys, how="left")
    agg.rename(columns={config.target_col: "actual_pts"}, inplace=True)
    agg["actual_ppg"] = (agg["actual_pts"] / agg["games"].clip(lower=1)).round(3)
    return agg


def score_holdout_season(weekly: pd.DataFrame, config: ProjectionConfig, season: int,
                          verbose: bool = False) -> pd.DataFrame | None:
    """Return a merged per-player frame with projected + actual points/VOR for `season`."""
    proj, _ = build_predictions_core(weekly, config, as_of_season=season)
    if proj.empty:
        return None
    proj = proj.copy()
    proj["proj_vor"] = proj.apply(
        lambda r: r["predicted_pts"] - REPLACEMENT_LEVEL_PPR.get(r[config.pos_col], 0), axis=1
    )

    actual = actual_outcomes_for_season(weekly, season, config)
    if actual.empty:
        return None
    baseline, debug = derive_replacement_baseline(
        actual, config.pos_col, "actual_pts", ROSTER_SLOTS, FLEX_ELIGIBLE, LEAGUE_SIZE,
        name_col=config.name_col,
    )
    if verbose:
        print(f"  [season {season}] dynamic actual-finish replacement baseline:")
        for pos, val in baseline.items():
            src = ", ".join(f"{n} ({p:.1f})" for n, p in debug.get(pos, []))
            print(f"    {pos}: {val:.1f} pts  (source: {src or 'n/a'})")

    actual = actual.copy()
    actual["actual_vor"] = actual.apply(
        lambda r: r["actual_pts"] - baseline.get(r[config.pos_col], 0), axis=1
    )

    key = [config.track_col, config.pos_col]
    merged = proj[key + [config.name_col, "predicted_pts", "pred_ppg", "proj_vor"]].merge(
        actual[key + ["actual_pts", "actual_ppg", "games", "actual_vor"]],
        on=key, how="inner",
    )
    merged["season"] = season
    return merged


def rank_within(df: pd.DataFrame, col: str, group_col: str | None = None) -> pd.Series:
    if group_col:
        return df.groupby(group_col)[col].rank(ascending=False, method="min")
    return df[col].rank(ascending=False, method="min")


def compute_metrics(merged: pd.DataFrame, pos_col: str) -> dict:
    out: dict = {}

    def _block(sub: pd.DataFrame) -> dict:
        if len(sub) < 3:
            return {"n": len(sub)}
        proj_rank = sub["proj_vor"].rank(ascending=False, method="min")
        act_rank = sub["actual_vor"].rank(ascending=False, method="min")
        rho, _ = spearmanr(proj_rank, act_rank)
        bias = (sub["predicted_pts"] - sub["actual_pts"])
        err = (sub["predicted_pts"] - sub["actual_pts"]).abs()
        vor_gap = (sub["proj_vor"] - sub["actual_vor"])
        block = {
            "n": len(sub),
            "spearman_vor_rank": None if pd.isna(rho) else round(float(rho), 4),
            "mean_abs_rank_error": round(float((proj_rank - act_rank).abs().mean()), 2),
            "mean_signed_bias_pts": round(float(bias.mean()), 2),
            "mean_abs_error_pts": round(float(err.mean()), 2),
            "rmse_pts": round(float(np.sqrt((err ** 2).mean())), 2),
            "mean_proj_vor_minus_actual_vor": round(float(vor_gap.mean()), 2),
        }
        return block

    overall_sorted = merged.copy()
    overall_sorted["proj_rank_overall"] = overall_sorted["proj_vor"].rank(ascending=False, method="min")
    overall_sorted["actual_rank_overall"] = overall_sorted["actual_vor"].rank(ascending=False, method="min")
    for n in TOP_N_HIT_RATES:
        if len(overall_sorted) < n:
            continue
        proj_top = set(overall_sorted.nsmallest(n, "proj_rank_overall").index)
        actual_top = set(overall_sorted.nsmallest(n, "actual_rank_overall").index)
        out[f"top{n}_hit_rate"] = round(len(proj_top & actual_top) / n, 3)

    out["overall"] = _block(merged)
    out["by_position"] = {pos: _block(sub) for pos, sub in merged.groupby(pos_col)}
    return out


def run_backtest(weekly: pd.DataFrame, config: ProjectionConfig, seasons: list[int],
                  verbose: bool = False) -> dict:
    frames = []
    per_season = {}
    for season in seasons:
        merged = score_holdout_season(weekly, config, season, verbose=verbose)
        if merged is None:
            print(f"  [season {season}] skipped — no data available before this season", file=sys.stderr)
            continue
        frames.append(merged)
        per_season[season] = compute_metrics(merged, config.pos_col)

    if not frames:
        raise SystemExit("No holdout seasons produced results — check weekly.csv coverage.")

    combined = pd.concat(frames, ignore_index=True)
    return {
        "config": {
            "target_col": config.target_col, "ridge_alpha": config.ridge_alpha,
            "decay": config.decay, "ppg_blend_weight": config.ppg_blend_weight,
            "ppg_baseline_games": config.ppg_baseline_games,
            "min_games_by_pos": config.min_games_by_pos,
        },
        "seasons": seasons,
        "per_season": per_season,
        "combined": compute_metrics(combined, config.pos_col),
    }


def print_report(results: dict, label: str = "") -> None:
    title = f"BACKTEST RESULTS {('— ' + label) if label else ''}"
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    print(f"Holdout seasons: {results['seasons']}")

    combined = results["combined"]
    print(f"\nTop-N hit rate (overall, pooled across all holdout seasons):")
    for n in TOP_N_HIT_RATES:
        key = f"top{n}_hit_rate"
        if key in combined:
            print(f"  Top-{n}: {combined[key]:.1%}")

    def _print_block(name: str, block: dict) -> None:
        if block.get("n", 0) < 3:
            print(f"  {name:>4}: n={block.get('n', 0)} (too few to score)")
            return
        print(
            f"  {name:>4}: n={block['n']:<4} "
            f"spearman={block['spearman_vor_rank']:<7} "
            f"mean_rank_err={block['mean_abs_rank_error']:<6} "
            f"bias={block['mean_signed_bias_pts']:>+7.1f} "
            f"MAE={block['mean_abs_error_pts']:<6.1f} "
            f"RMSE={block['rmse_pts']:<6.1f} "
            f"VOR_gap={block['mean_proj_vor_minus_actual_vor']:>+7.1f}"
        )

    print("\nCombined across all holdout seasons (primary metric = spearman):")
    _print_block("ALL", combined["overall"])
    for pos, block in combined["by_position"].items():
        _print_block(pos, block)

    print("\nPer-season breakdown (overall row only):")
    for season, metrics in results["per_season"].items():
        _print_block(str(season), metrics["overall"])


def load_config_overrides(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_config(name_col: str, pos_col: str, team_col: str | None, track_col: str,
                  overrides: dict | None = None) -> ProjectionConfig:
    config = ProjectionConfig(
        target_col="fantasy_points_ppr", name_col=name_col, pos_col=pos_col,
        team_col=team_col, track_col=track_col,
    )
    for key, value in (overrides or {}).items():
        setattr(config, key, value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, help="Single holdout season (shorthand for --seasons S).")
    parser.add_argument("--seasons", type=int, nargs="+", default=None,
                         help="Holdout seasons to walk forward over (default: 2021-2025).")
    parser.add_argument("--output", type=str, default=None, help="Write results JSON to this path.")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("CONFIG_A", "CONFIG_B"),
                         help="Two JSON files of ProjectionConfig field overrides; runs both and diffs.")
    parser.add_argument("--verbose", action="store_true", help="Log dynamic replacement-baseline source players.")
    args = parser.parse_args()

    if args.season and args.seasons:
        raise SystemExit("Pass either --season or --seasons, not both.")
    seasons = [args.season] if args.season else (args.seasons or [2021, 2022, 2023, 2024, 2025])

    weekly = load_weekly(REPO_ROOT)
    name_col, pos_col, team_col, track_col = resolve_columns(weekly)

    if args.compare:
        results = {}
        for label, path in zip(("A", "B"), args.compare):
            config = build_config(name_col, pos_col, team_col, track_col, load_config_overrides(path))
            print(f"\nRunning config {label} ({path}) ...")
            results[label] = run_backtest(weekly, config, seasons, verbose=args.verbose)
            print_report(results[label], label=f"config {label} ({path})")

        print(f"\n{'=' * 40}\nCOMPARE: combined-overall spearman A vs B\n{'=' * 40}")
        a_overall = results["A"]["combined"]["overall"]
        b_overall = results["B"]["combined"]["overall"]
        print(f"  A: {a_overall['spearman_vor_rank']}   B: {b_overall['spearman_vor_rank']}   "
              f"delta: {round(b_overall['spearman_vor_rank'] - a_overall['spearman_vor_rank'], 4)}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nSaved comparison results to {args.output}")
        return

    config = build_config(name_col, pos_col, team_col, track_col)
    results = run_backtest(weekly, config, seasons, verbose=args.verbose)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
