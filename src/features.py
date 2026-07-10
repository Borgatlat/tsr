"""
Build match-level features with exponential recency weighting.

Run:
    python src/features.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURES_PATH,
    FIFA_RAW,
    MATCHES_FILTERED,
    ensure_dirs,
    normalize_team,
)


def _result_points(goals_for: int, goals_against: int) -> float:
    """3 = win, 1 = draw, 0 = loss — standard league points."""
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _win_rate_from_points(points: list[float]) -> float:
    if not points:
        return 0.5
    return np.mean([p / 3.0 for p in points])


def _team_form(team: str, history: list[dict], n: int = 20) -> dict:
    """Rolling win rate and goal averages from a team's prior matches."""
    recent = history[-n:]
    if not recent:
        return {
            "win_rate": 0.5,
            "avg_goals_scored": 1.2,
            "avg_goals_conceded": 1.2,
        }

    points = [_result_points(m["gf"], m["ga"]) for m in recent]
    return {
        "win_rate": _win_rate_from_points(points),
        "avg_goals_scored": float(np.mean([m["gf"] for m in recent])),
        "avg_goals_conceded": float(np.mean([m["ga"] for m in recent])),
    }


def _h2h_team_win_rate(team: str, opponent: str, h2h: dict) -> float:
    """A team's win rate vs an opponent, regardless of who was listed as home."""
    key = tuple(sorted((team, opponent)))
    meetings = h2h.get(key, [])
    if not meetings:
        return 0.5

    wins = sum(
        1
        for m in meetings
        if (m["home"] == team and m["home_score"] > m["away_score"])
        or (m["away"] == team and m["away_score"] > m["home_score"])
    )
    return wins / len(meetings)


def _h2h_home_rate(home: str, away: str, h2h: dict) -> float:
    """Home team's historical win rate against this opponent (excluding current match)."""
    key = tuple(sorted((home, away)))
    meetings = h2h.get(key, [])
    if not meetings:
        return 0.5

    home_wins = sum(
        1
        for m in meetings
        if (m["home"] == home and m["home_score"] > m["away_score"])
        or (m["away"] == home and m["away_score"] > m["home_score"])
    )
    return home_wins / len(meetings)


def _lookup_fifa_gap(
    home: str,
    away: str,
    match_date: pd.Timestamp,
    rankings: pd.DataFrame,
) -> float:
    """
    FIFA ranking gap = home_rank - away_rank.
    Negative gap means home team is ranked higher (better).
    """
    prior = rankings[rankings["date"] <= match_date]
    if prior.empty:
        return 0.0

    latest_date = prior["date"].max()
    snapshot = prior[prior["date"] == latest_date].set_index("team")

    home_rank = snapshot.loc[home, "rank"] if home in snapshot.index else 100.0
    away_rank = snapshot.loc[away, "rank"] if away in snapshot.index else 100.0
    return float(home_rank - away_rank)


def _recency_weight(match_date: pd.Timestamp, reference: datetime | None = None) -> float:
    """Exponential decay: weight = 0.85 ** years_ago."""
    ref = reference or datetime.now()
    years_ago = max((ref - match_date.to_pydatetime()).days / 365.25, 0.0)
    return 0.85 ** years_ago


def _neutral_feature_dict(
    team_a: str,
    team_b: str,
    team_a_form: dict,
    team_b_form: dict,
    h2h_history: dict,
    rankings: pd.DataFrame,
    match_date: pd.Timestamp,
) -> dict:
    """
    Symmetric-friendly features for neutral venues.
    team_a is the first-named side (e.g. Spain in 'Spain vs France').
    """
    h2h_a = _h2h_team_win_rate(team_a, team_b, h2h_history)
    h2h_b = _h2h_team_win_rate(team_b, team_a, h2h_history)
    gap = _lookup_fifa_gap(team_a, team_b, match_date, rankings)

    return {
        "team_a_win_rate_20": team_a_form["win_rate"],
        "team_b_win_rate_20": team_b_form["win_rate"],
        "win_rate_diff": team_a_form["win_rate"] - team_b_form["win_rate"],
        "team_a_avg_goals_scored": team_a_form["avg_goals_scored"],
        "team_b_avg_goals_scored": team_b_form["avg_goals_scored"],
        "goals_scored_diff": team_a_form["avg_goals_scored"] - team_b_form["avg_goals_scored"],
        "team_a_avg_goals_conceded": team_a_form["avg_goals_conceded"],
        "team_b_avg_goals_conceded": team_b_form["avg_goals_conceded"],
        "goals_conceded_diff": team_a_form["avg_goals_conceded"] - team_b_form["avg_goals_conceded"],
        "h2h_team_a_win_rate": h2h_a,
        "h2h_team_b_win_rate": h2h_b,
        "fifa_ranking_gap": gap,
        "abs_fifa_ranking_gap": abs(gap),
    }


NEUTRAL_OUTCOME_FEATURES = [
    "team_a_win_rate_20",
    "team_b_win_rate_20",
    "win_rate_diff",
    "team_a_avg_goals_scored",
    "team_b_avg_goals_scored",
    "goals_scored_diff",
    "team_a_avg_goals_conceded",
    "team_b_avg_goals_conceded",
    "goals_conceded_diff",
    "h2h_team_a_win_rate",
    "h2h_team_b_win_rate",
    "fifa_ranking_gap",
    "abs_fifa_ranking_gap",
]


def _outcome_label_team_a(team_a_score: int, team_b_score: int) -> int:
    """0 = team_a win, 1 = draw, 2 = team_b win."""
    if team_a_score > team_b_score:
        return 0
    if team_a_score == team_b_score:
        return 1
    return 2


def build_neutral_features(
    team_a: str,
    team_b: str,
    team_history: dict[str, list[dict]] | None = None,
    h2h_history: dict[tuple[str, str], list[dict]] | None = None,
    rankings: pd.DataFrame | None = None,
    match_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build neutral-venue feature row for team_a vs team_b."""
    team_a = normalize_team(team_a)
    team_b = normalize_team(team_b)
    match_date = match_date or pd.Timestamp.now()

    if team_history is None or h2h_history is None:
        raise ValueError("team_history and h2h_history are required")

    if rankings is None:
        rankings = pd.read_csv(FIFA_RAW, parse_dates=["date"])
        rankings["team"] = rankings["team"].map(normalize_team)

    team_a_form = _team_form(team_a, team_history.get(team_a, []))
    team_b_form = _team_form(team_b, team_history.get(team_b, []))
    features = _neutral_feature_dict(
        team_a, team_b, team_a_form, team_b_form, h2h_history, rankings, match_date
    )
    return pd.DataFrame([features])


def _outcome_label(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def build_features(
    matches: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Construct the full feature matrix from chronological match history."""
    ensure_dirs()

    if matches is None:
        matches = pd.read_csv(MATCHES_FILTERED, parse_dates=["date"])
    if rankings is None:
        rankings = pd.read_csv(FIFA_RAW, parse_dates=["date"])
        rankings["team"] = rankings["team"].map(normalize_team)

    matches = matches.sort_values("date").reset_index(drop=True)
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)

    team_history: dict[str, list[dict]] = {}
    h2h_history: dict[tuple[str, str], list[dict]] = {}
    rows: list[dict] = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        home_form = _team_form(home, team_history.get(home, []))
        away_form = _team_form(away, team_history.get(away, []))
        h2h_rate = _h2h_home_rate(home, away, h2h_history)
        ranking_gap = _lookup_fifa_gap(home, away, match_date, rankings)
        weight = _recency_weight(match_date, reference_date)
        neutral_feats = _neutral_feature_dict(
            home, away, home_form, away_form, h2h_history, rankings, match_date
        )

        row = {
                "date": match_date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": row.get("tournament", ""),
                "neutral": row.get("neutral", False),
                "home_win_rate_20": home_form["win_rate"],
                "away_win_rate_20": away_form["win_rate"],
                "home_avg_goals_scored": home_form["avg_goals_scored"],
                "home_avg_goals_conceded": home_form["avg_goals_conceded"],
                "away_avg_goals_scored": away_form["avg_goals_scored"],
                "away_avg_goals_conceded": away_form["avg_goals_conceded"],
                "h2h_home_win_rate": h2h_rate,
                "fifa_ranking_gap": ranking_gap,
                "recency_weight": weight,
                "outcome": _outcome_label(home_score, away_score),
                "team_a_outcome": _outcome_label_team_a(home_score, away_score),
            }
        row.update(neutral_feats)
        rows.append(row)

        # Update histories AFTER feature snapshot (no leakage).
        team_history.setdefault(home, []).append(
            {"gf": home_score, "ga": away_score, "date": match_date}
        )
        team_history.setdefault(away, []).append(
            {"gf": away_score, "ga": home_score, "date": match_date}
        )
        key = tuple(sorted((home, away)))
        h2h_history.setdefault(key, []).append(
            {
                "home": home,
                "away": away,
                "home_score": home_score,
                "away_score": away_score,
                "date": match_date,
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(FEATURES_PATH, index=False)
    return features


def main() -> None:
    features = build_features()
    print(f"Built features for {len(features):,} matches")
    print("\nSample rows (with recency weights):")
    sample_cols = [
        "date",
        "home_team",
        "away_team",
        "home_win_rate_20",
        "away_win_rate_20",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "h2h_home_win_rate",
        "fifa_ranking_gap",
        "recency_weight",
        "outcome",
    ]
    print(features[sample_cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
