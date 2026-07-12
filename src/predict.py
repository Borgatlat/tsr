"""
Predict win/draw/loss probabilities and expected goals for a matchup.

Run from project root:
    python predict.py Spain Belgium
    python predict.py Spain France --neutral

Or:
    python src/predict.py Spain Belgium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running as `python src/predict.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FIFA_RAW,
    GOALS_MODEL_PATH,
    MATCHES_FILTERED,
    NEUTRAL_OUTCOME_MODEL_PATH,
    OUTCOME_MODEL_PATH,
    normalize_team,
)
from features import (
    NEUTRAL_OUTCOME_FEATURES,
    _h2h_home_rate,
    _lookup_fifa_gap,
    _team_form,
    build_neutral_features,
)
from train_model import OUTCOME_FEATURES, OUTCOME_LABELS


def _load_histories(
    inject: list[tuple[str, str, int, int]] | None = None,
) -> tuple[dict, dict]:
    """Rebuild team and head-to-head histories from all stored matches."""
    matches = pd.read_csv(MATCHES_FILTERED, parse_dates=["date"]).sort_values("date")
    matches = matches.dropna(subset=["home_score", "away_score"])
    team_history: dict[str, list] = {}
    h2h_history: dict[tuple, list] = {}

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])

        team_history.setdefault(home, []).append({"gf": hs, "ga": as_})
        team_history.setdefault(away, []).append({"gf": as_, "ga": hs})

        key = tuple(sorted((home, away)))
        h2h_history.setdefault(key, []).append(
            {
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": as_,
            }
        )

    for entry in inject or []:
        team_a, team_b, goals_a, goals_b = entry
        team_a, team_b = normalize_team(team_a), normalize_team(team_b)
        team_history.setdefault(team_a, []).append({"gf": goals_a, "ga": goals_b})
        team_history.setdefault(team_b, []).append({"gf": goals_b, "ga": goals_a})
        key = tuple(sorted((team_a, team_b)))
        h2h_history.setdefault(key, []).append(
            {
                "home": team_a,
                "away": team_b,
                "home_score": goals_a,
                "away_score": goals_b,
            }
        )

    return team_history, h2h_history


def build_match_features(
    home: str,
    away: str,
    histories: tuple[dict, dict] | None = None,
) -> pd.DataFrame:
    """Assemble feature vector for a home-vs-away fixture."""
    home = normalize_team(home)
    away = normalize_team(away)

    if histories is None:
        team_history, h2h_history = _load_histories()
    else:
        team_history, h2h_history = histories
    rankings = pd.read_csv(FIFA_RAW, parse_dates=["date"])
    rankings["team"] = rankings["team"].map(normalize_team)

    home_form = _team_form(home, team_history.get(home, []))
    away_form = _team_form(away, team_history.get(away, []))
    h2h = _h2h_home_rate(home, away, h2h_history)
    gap = _lookup_fifa_gap(home, away, pd.Timestamp.now(), rankings)

    return pd.DataFrame(
        [
            {
                "home_win_rate_20": home_form["win_rate"],
                "away_win_rate_20": away_form["win_rate"],
                "home_avg_goals_scored": home_form["avg_goals_scored"],
                "home_avg_goals_conceded": home_form["avg_goals_conceded"],
                "away_avg_goals_scored": away_form["avg_goals_scored"],
                "away_avg_goals_conceded": away_form["avg_goals_conceded"],
                "h2h_home_win_rate": h2h,
                "fifa_ranking_gap": gap,
            }
        ]
    )


def _neutral_prediction(
    outcome_model,
    goals_model,
    team_a: str,
    team_b: str,
    histories: tuple[dict, dict] | None = None,
) -> dict:
    """Single-pass neutral prediction for team_a vs team_b."""
    if histories is None:
        team_history, h2h_history = _load_histories()
    else:
        team_history, h2h_history = histories
    features = build_neutral_features(team_a, team_b, team_history, h2h_history)
    probs = outcome_model.predict_proba(features[NEUTRAL_OUTCOME_FEATURES])[0]
    pred_class = int(np.argmax(probs))
    outcome_labels = {0: f"{team_a} win", 1: "Draw", 2: f"{team_b} win"}

    team_a_row = pd.DataFrame([{"team": team_a, "opponent": team_b, "is_home": 0}])
    team_b_row = pd.DataFrame([{"team": team_b, "opponent": team_a, "is_home": 0}])
    pred_a_goals = float(goals_model.predict(team_a_row)[0])
    pred_b_goals = float(goals_model.predict(team_b_row)[0])

    return {
        "team_a": team_a,
        "team_b": team_b,
        "neutral": True,
        "team_a_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "team_b_win_prob": float(probs[2]),
        "predicted_outcome": outcome_labels[pred_class],
        "predicted_team_a_goals": pred_a_goals,
        "predicted_team_b_goals": pred_b_goals,
    }


def predict_match(
    home: str,
    away: str,
    neutral: bool = False,
    recent_results: list[tuple[str, str, int, int]] | None = None,
) -> dict:
    """Return outcome probabilities and expected goals for a matchup."""
    if not OUTCOME_MODEL_PATH.exists() or not GOALS_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Run: python src/train_model.py"
        )

    goals_model = joblib.load(GOALS_MODEL_PATH)
    team_a = normalize_team(home)
    team_b = normalize_team(away)
    histories = _load_histories(inject=recent_results)

    if neutral:
        if not NEUTRAL_OUTCOME_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Neutral model not found. Run: python src/train_model.py"
            )
        outcome_model = joblib.load(NEUTRAL_OUTCOME_MODEL_PATH)

        # Average both orderings so argument order never affects the result.
        ab = _neutral_prediction(outcome_model, goals_model, team_a, team_b, histories)
        ba = _neutral_prediction(outcome_model, goals_model, team_b, team_a, histories)

        team_a_win = (ab["team_a_win_prob"] + ba["team_b_win_prob"]) / 2
        team_b_win = (ab["team_b_win_prob"] + ba["team_a_win_prob"]) / 2
        draw = (ab["draw_prob"] + ba["draw_prob"]) / 2
        pred_a_goals = (ab["predicted_team_a_goals"] + ba["predicted_team_b_goals"]) / 2
        pred_b_goals = (ab["predicted_team_b_goals"] + ba["predicted_team_a_goals"]) / 2

        probs = [team_a_win, draw, team_b_win]
        pred_class = int(np.argmax(probs))
        outcome_labels = {0: f"{team_a} win", 1: "Draw", 2: f"{team_b} win"}

        return {
            "team_a": team_a,
            "team_b": team_b,
            "neutral": True,
            "team_a_win_prob": float(team_a_win),
            "draw_prob": float(draw),
            "team_b_win_prob": float(team_b_win),
            "predicted_outcome": outcome_labels[pred_class],
            "predicted_team_a_goals": float(pred_a_goals),
            "predicted_team_b_goals": float(pred_b_goals),
        }

    return _predict_home_away(team_a, team_b, goals_model, histories)


def _predict_home_away(
    team_a: str,
    team_b: str,
    goals_model,
    histories: tuple[dict, dict] | None = None,
) -> dict:
    """Predict a true home-vs-away fixture (team_a at home)."""
    outcome_model = joblib.load(OUTCOME_MODEL_PATH)
    features = build_match_features(team_a, team_b, histories)
    probs = outcome_model.predict_proba(features[OUTCOME_FEATURES])[0]
    pred_class = int(np.argmax(probs))

    team_a_row = pd.DataFrame([{"team": team_a, "opponent": team_b, "is_home": 1}])
    team_b_row = pd.DataFrame([{"team": team_b, "opponent": team_a, "is_home": 0}])
    pred_a_goals = float(goals_model.predict(team_a_row)[0])
    pred_b_goals = float(goals_model.predict(team_b_row)[0])

    return {
        "team_a": team_a,
        "team_b": team_b,
        "neutral": False,
        "home_team": team_a,
        "away_team": team_b,
        "home_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[2]),
        "predicted_outcome": OUTCOME_LABELS[pred_class],
        "predicted_home_goals": pred_a_goals,
        "predicted_away_goals": pred_b_goals,
    }


def _knockout_probs(
    team_a_win: float, draw: float, team_b_win: float
) -> tuple[float, float, float]:
    """
    Redistribute win probabilities as if a draw cannot stand (ET + penalties).
    Removes draw mass and rescales so team_a + team_b = 100%.
    """
    non_draw = team_a_win + team_b_win
    if non_draw <= 0:
        return 0.5, draw, 0.5
    return team_a_win / non_draw, draw, team_b_win / non_draw


def print_prediction(result: dict, knockout: bool = False, note: str | None = None) -> None:
    """Pretty-print prediction output."""
    if note:
        print(f"Context: {note}")
    if result.get("neutral"):
        print(f"\n{result['team_a']} vs {result['team_b']}  (neutral venue)")
        print("-" * 40)
        a_win, draw, b_win = (
            result["team_a_win_prob"],
            result["draw_prob"],
            result["team_b_win_prob"],
        )
        if knockout:
            a_win, _, b_win = _knockout_probs(a_win, draw, b_win)
            print("Knockout mode (draw removed — ET/penalties decide winner)")
            print(
                f"{result['team_a']} to advance: {a_win:.1%} | "
                f"{result['team_b']} to advance: {b_win:.1%}"
            )
        else:
            print(
                f"{result['team_a']} win: {a_win:.1%} | "
                f"Draw: {draw:.1%} | "
                f"{result['team_b']} win: {b_win:.1%}"
            )
            print(f"Most likely outcome: {result['predicted_outcome']}")
        print(
            f"Predicted score (90 min): {result['predicted_team_a_goals']:.2f} - "
            f"{result['predicted_team_b_goals']:.2f}"
        )
        return

    print(f"\n{result['home_team']} vs {result['away_team']}  (home vs away)")
    print("-" * 40)
    home_win, draw, away_win = (
        result["home_win_prob"],
        result["draw_prob"],
        result["away_win_prob"],
    )
    if knockout:
        home_win, _, away_win = _knockout_probs(home_win, draw, away_win)
        print("Knockout mode (draw removed — ET/penalties decide winner)")
        print(
            f"{result['home_team']} to advance: {home_win:.1%} | "
            f"{result['away_team']} to advance: {away_win:.1%}"
        )
        fav = result["home_team"] if home_win >= away_win else result["away_team"]
        print(f"Favorite to advance: {fav}")
    else:
        print(
            f"Home win: {home_win:.1%} | "
            f"Draw: {draw:.1%} | "
            f"Away win: {away_win:.1%}"
        )
        print(f"Most likely outcome: {result['predicted_outcome']}")
    print(
        f"Predicted score (90 min): {result['predicted_home_goals']:.2f} - "
        f"{result['predicted_away_goals']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict match outcome and goals for two national teams"
    )
    parser.add_argument("home_team", help="First team (e.g. Spain)")
    parser.add_argument("away_team", help="Second team (e.g. Belgium)")
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Neutral venue — uses dedicated neutral outcome model",
    )
    parser.add_argument(
        "--knockout",
        action="store_true",
        help="Knockout tiebreaker — remove draw; show who advances after ET/penalties",
    )
    parser.add_argument(
        "--after",
        nargs=3,
        metavar=("OPPONENT", "GOALS_FOR", "GOALS_AGAINST"),
        help="Recent result for the first team (e.g. --after Switzerland 3 1)",
    )
    parser.add_argument(
        "--after-away",
        nargs=3,
        metavar=("OPPONENT", "GOALS_FOR", "GOALS_AGAINST"),
        help="Recent result for the second team (e.g. --after-away Norway 2 1)",
    )
    args = parser.parse_args()

    recent: list[tuple[str, str, int, int]] = []
    notes: list[str] = []
    if args.after:
        opponent, goals_for, goals_against = args.after
        recent.append((args.home_team, opponent, int(goals_for), int(goals_against)))
        notes.append(
            f"{args.home_team} beat {opponent} {goals_for}-{goals_against} (ET)"
        )
    if args.after_away:
        opponent, goals_for, goals_against = args.after_away
        recent.append((args.away_team, opponent, int(goals_for), int(goals_against)))
        notes.append(
            f"{args.away_team} beat {opponent} {goals_for}-{goals_against} (ET)"
        )

    note = " | ".join(notes) if notes else None

    result = predict_match(
        args.home_team,
        args.away_team,
        neutral=args.neutral,
        recent_results=recent or None,
    )
    print_prediction(result, knockout=args.knockout, note=note)


if __name__ == "__main__":
    main()
