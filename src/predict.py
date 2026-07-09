"""
Predict win/draw/loss probabilities and expected goals for a matchup.

Run from project root:
    python predict.py Spain Belgium

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
    OUTCOME_MODEL_PATH,
    normalize_team,
)
from features import _h2h_home_rate, _lookup_fifa_gap, _team_form
from train_model import OUTCOME_FEATURES, OUTCOME_LABELS


def _load_histories() -> tuple[dict, dict]:
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
    return team_history, h2h_history


def build_match_features(home: str, away: str) -> pd.DataFrame:
    """Assemble feature vector for a hypothetical home-vs-away fixture today."""
    home = normalize_team(home)
    away = normalize_team(away)

    team_history, h2h_history = _load_histories()
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


def predict_match(home: str, away: str) -> dict:
    """Return outcome probabilities and expected goals for home vs away."""
    if not OUTCOME_MODEL_PATH.exists() or not GOALS_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. Run: python src/train_model.py"
        )

    outcome_model = joblib.load(OUTCOME_MODEL_PATH)
    goals_model = joblib.load(GOALS_MODEL_PATH)

    home = normalize_team(home)
    away = normalize_team(away)
    features = build_match_features(home, away)

    probs = outcome_model.predict_proba(features[OUTCOME_FEATURES])[0]
    pred_class = int(np.argmax(probs))

    home_row = pd.DataFrame([{"team": home, "opponent": away, "is_home": 1}])
    away_row = pd.DataFrame([{"team": away, "opponent": home, "is_home": 0}])
    pred_home = float(goals_model.predict(home_row)[0])
    pred_away = float(goals_model.predict(away_row)[0])

    return {
        "home_team": home,
        "away_team": away,
        "home_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[2]),
        "predicted_outcome": OUTCOME_LABELS[pred_class],
        "predicted_home_goals": pred_home,
        "predicted_away_goals": pred_away,
    }


def print_prediction(result: dict) -> None:
    """Pretty-print prediction output."""
    print(f"\n{result['home_team']} vs {result['away_team']}")
    print("-" * 40)
    print(
        f"Home win: {result['home_win_prob']:.1%} | "
        f"Draw: {result['draw_prob']:.1%} | "
        f"Away win: {result['away_win_prob']:.1%}"
    )
    print(f"Most likely outcome: {result['predicted_outcome']}")
    print(
        f"Predicted score: {result['predicted_home_goals']:.2f} - "
        f"{result['predicted_away_goals']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict match outcome and goals for two national teams"
    )
    parser.add_argument("home_team", help="Home team name (e.g. Spain)")
    parser.add_argument("away_team", help="Away team name (e.g. Belgium)")
    args = parser.parse_args()

    result = predict_match(args.home_team, args.away_team)
    print_prediction(result)


if __name__ == "__main__":
    main()
