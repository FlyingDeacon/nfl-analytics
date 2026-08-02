"""Linking ESPN's draft feed to the big board.

The two systems key players differently: ESPN speaks in numeric playerIds and
its own fullName spellings, while the big board is keyed on nflverse display
names. Bridging them is the whole risk of syncing a live draft — a pick we fail
to resolve leaves that player sitting "available" on the board, and the
suggestion engine will cheerfully recommend someone who is already gone.

So the matching runs *before* the draft rather than during it, and it reports
what it couldn't link instead of silently dropping it. The direction that
matters is board -> ESPN: an ESPN player with no board row is just a late-round
flier outside our top 400 (harmless), but a *board* player with no ESPN id is
someone we can never mark as drafted.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

# ESPN's position labels -> the board's.
_POS_ALIASES = {"D/ST": "DEF", "DST": "DEF", "PK": "K"}

# Team abbreviation differences between the two feeds.
_TEAM_ALIASES = {"LA": "LAR", "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI"}

# Hand-maintained links for players the cascade below can't reach on its own —
# usually a nickname one feed uses and the other doesn't. Board name -> ESPN
# fullName. Nicknames defeat fuzzy matching by design: "Marquise"/"Hollywood"
# share no letters, so nothing short of a lookup will connect them.
NAME_ALIASES: dict[str, str] = {
    "Marquise Brown": "Hollywood Brown",
    "Kenneth Gainwell": "Kenny Gainwell",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# ESPN marks an unfilled draft slot with playerId -1. The test has to be an
# exact match, not "positive": team defenses carry *negative* ids (-16001 and
# down), so a `> 0` check would silently discard every D/ST pick and leave
# defenses sitting on the board as available for the whole draft.
_EMPTY_SLOT = -1


def pick_made(pick: dict) -> bool:
    """True if an ESPN draft slot has actually been filled."""
    pid = pick.get("playerId")
    return pid is not None and pid != _EMPTY_SLOT

# Similarity at or above this gets reported as a near miss worth a human's
# eye. Nothing is ever linked on similarity alone — see the fuzzy pass in
# resolve_players() for why.
_SUSPECT_MIN = 0.70


def normalize_name(name: str) -> str:
    """Lowercased, de-accented, punctuation- and suffix-free form for matching.

    Collapses the spellings that actually differ between the feeds: accents
    ("Amon-Ra St. Brown"), punctuation ("A.J." vs "AJ", "Ka'imi"), and generational
    suffixes ("Marvin Harrison Jr." vs "Marvin Harrison").
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower())
    parts = s.split()
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def normalize_pos(pos: str) -> str:
    return _POS_ALIASES.get(str(pos).upper().strip(), str(pos).upper().strip())


def normalize_team(team: str) -> str:
    return _TEAM_ALIASES.get(str(team).upper().strip(), str(team).upper().strip())


def _last_name(norm: str) -> str:
    parts = norm.split()
    return parts[-1] if parts else ""


def _first_name(norm: str) -> str:
    parts = norm.split()
    return parts[0] if parts else ""


def _prefix_compatible(a: str, b: str, min_len: int = 3) -> bool:
    """True when one first name is just a truncation of the other.

    Josh/Joshua and Rob/Robert qualify; Kenny/Kenneth and Omar/Amari do not.
    Deliberately stricter than a similarity score, which rates all four alike.
    """
    if not a or not b or min(len(a), len(b)) < min_len:
        return False
    return a.startswith(b) or b.startswith(a)


def resolve_players(board: pd.DataFrame, players: pd.DataFrame,
                    aliases: dict | None = None) -> pd.DataFrame:
    """Link every big-board player to an ESPN playerId.

    Returns one row per board player with the id it resolved to (or nulls when
    it didn't), plus how the link was made:

      exact     name + position agree
      alias     hand-maintained NAME_ALIASES entry
      name      names agree but ESPN lists a different position
      def-team  a D/ST matched on team rather than nickname
      fuzzy     same position, surname and team, first name only truncated
      (blank)   unresolved — see the `suspect` column for the nearest miss

    Pure pandas, no Streamlit, so it can be exercised from a script.
    """
    aliases = {**NAME_ALIASES, **(aliases or {})}
    cols = ["player", "pos", "team", "player_id", "espn_name", "espn_rank",
            "method", "score", "suspect"]
    if board.empty or players.empty:
        return pd.DataFrame(columns=cols)

    pool = players[players["player_id"].notna()].copy()
    pool["nkey"] = pool["player"].map(normalize_name)
    pool["pkey"] = pool["pos"].map(normalize_pos)
    pool["tkey"] = pool["nfl_team"].map(normalize_team)

    # Lower ESPN rank = more likely the intended player, so keep the best-ranked
    # claimant whenever two ESPN entries normalize to the same name.
    pool = pool.sort_values("espn_rank", na_position="last")

    by_name_pos: dict[tuple, dict] = {}
    by_name: dict[str, dict] = {}
    by_def_team: dict[str, dict] = {}
    for rec in pool.to_dict("records"):
        by_name_pos.setdefault((rec["nkey"], rec["pkey"]), rec)
        by_name.setdefault(rec["nkey"], rec)
        if rec["pkey"] == "DEF":
            by_def_team.setdefault(rec["tkey"], rec)

    pool_records = pool.to_dict("records")
    taken: set = set()
    rows = []

    for b in board.to_dict("records"):
        nkey = normalize_name(b["player"])
        pkey = normalize_pos(b["pos"])
        tkey = normalize_team(b.get("team", ""))

        hit = method = None
        if b["player"] in aliases:
            hit = by_name.get(normalize_name(aliases[b["player"]]))
            method = "alias" if hit else None
        if hit is None and (nkey, pkey) in by_name_pos:
            hit, method = by_name_pos[(nkey, pkey)], "exact"
        if hit is None and nkey in by_name:
            hit, method = by_name[nkey], "name"
        if hit is None and pkey == "DEF" and tkey in by_def_team:
            hit, method = by_def_team[tkey], "def-team"

        score, suspect = 1.0 if hit is not None else None, ""
        if hit is None:
            # Fuzzy pass, and deliberately a narrow one. A wrong auto-link is far
            # worse than an honest gap: the gap gets reported and fixed, whereas a
            # wrong link silently marks the wrong player drafted and poisons every
            # suggestion after it. Similarity alone can't tell "Josh"/"Joshua"
            # (same guy, 0.80) from "Omar"/"Amari" Cooper (two players, 0.87), so
            # it never decides anything — it only ranks what to show a human.
            # A link requires position, surname and team to agree outright, with
            # the first names differing by no more than truncation.
            last, first = _last_name(nkey), _first_name(nkey)
            best, best_score = None, 0.0
            for rec in pool_records:
                if rec["pkey"] != pkey:
                    continue
                if rec["tkey"] != tkey and _last_name(rec["nkey"]) != last:
                    continue
                r = SequenceMatcher(None, nkey, rec["nkey"]).ratio()
                if r > best_score:
                    best, best_score = rec, r
            if best is not None:
                safe = (best["tkey"] == tkey
                        and _last_name(best["nkey"]) == last
                        and _prefix_compatible(first, _first_name(best["nkey"])))
                if safe:
                    hit, method, score = best, "fuzzy", round(best_score, 3)
                elif best_score >= _SUSPECT_MIN:
                    suspect = (f"looks like {best['player']} ({best['pkey']} "
                               f"{best['tkey']}, {best_score:.2f}) — add to "
                               "NAME_ALIASES if that's him")

        pid = hit["player_id"] if hit is not None else None
        if pid is not None and pid in taken:
            # Two board rows claiming one ESPN id means one of them is wrong;
            # leave the later claimant unresolved rather than double-linking.
            suspect = f"id {int(pid)} already linked to another board player"
            hit = pid = method = score = None

        if pid is not None:
            taken.add(pid)

        rows.append({
            "player": b["player"], "pos": b["pos"], "team": b.get("team", ""),
            "player_id": pid,
            "espn_name": hit["player"] if hit is not None else "",
            "espn_rank": hit.get("espn_rank") if hit is not None else None,
            "method": method or "", "score": score, "suspect": suspect,
        })

    return pd.DataFrame(rows, columns=cols)


def id_to_player(links: pd.DataFrame) -> dict:
    """ESPN playerId -> big-board player name, for replaying a draft feed."""
    ok = links[links["player_id"].notna()]
    return {int(pid): name for pid, name in zip(ok["player_id"], ok["player"])}


def replay(picks: list, links: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    """Rebuild draft state from an ESPN picks array, annotated with board value.

    Rebuild rather than append: ESPN hands over the whole draft on every poll,
    picks can arrive out of order, a poll can be missed, and a commissioner can
    undo. Replaying a couple hundred picks costs nothing and can never drift.

    ESPN pre-populates the whole board before anyone picks: every slot is
    already there with its teamId assigned and playerId -1. So the length of
    the array is the size of the draft, not its progress — only entries with a
    real playerId count as picks made.

    `on_board` False means the pick was a flier outside our top few hundred —
    normal and harmless. It is *not* the same as a resolver miss, which is
    caught up front by resolve_players() and would be a silent correctness bug.
    """
    cols = ["overall", "round", "pick", "team_id", "player_id", "keeper",
            "on_board", "player", "pos", "team", "my_rank", "vor"]
    picks = [p for p in (picks or []) if pick_made(p)]
    if not picks:
        return pd.DataFrame(columns=cols)

    id_map = id_to_player(links)
    info = board.set_index("player") if not board.empty else pd.DataFrame()
    ranks = dict(zip(board["player"], board.get("my_rank", range(1, len(board) + 1))))

    rows = []
    for p in sorted(picks, key=lambda x: x.get("overallPickNumber") or 0):
        pid = p.get("playerId")
        name = id_map.get(int(pid)) if pid is not None else None
        row = {
            "overall": p.get("overallPickNumber"), "round": p.get("roundId"),
            "pick": p.get("roundPickNumber"), "team_id": p.get("teamId"),
            "player_id": pid, "keeper": bool(p.get("keeper")),
            "on_board": name is not None, "player": name or "",
            "pos": "", "team": "", "my_rank": None, "vor": None,
        }
        if name is not None and name in info.index:
            b = info.loc[name]
            row |= {"pos": b["pos"], "team": b.get("team", ""),
                    "my_rank": ranks.get(name), "vor": round(float(b["vor"]), 1)}
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def available(board: pd.DataFrame, replayed: pd.DataFrame) -> pd.DataFrame:
    """Board rows nobody has drafted yet, best first."""
    gone = set(replayed.loc[replayed["on_board"], "player"]) if not replayed.empty else set()
    return board[~board["player"].isin(gone)]
