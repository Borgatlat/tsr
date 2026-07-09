"""
Train outcome and goals models, then backtest on 2022 World Cup group stage.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURES_PATH,
    GOALS_ENCODERS_PATH,
    GOALS_MODEL_PATH,
    MATCHES_FILTERED,
    OUTCOME_MODEL_PATH,
    ensure_dirs,
)
from features import build_features

OUTCOME_FEATURES = [
    "home_win_rate_20",
    "away_win_rate_20",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "h2h_home_win_rate",
    "fifa_ranking_gap",
]

OUTCOME_LABELS = {0: "Home win", 1: "Draw", 2: "Away win"}


def _load_or_build_features() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    return build_features()


def _split_train_holdout(features: pd.DataFrame, holdout_start: str) -> tuple:
    """Train on matches before holdout_start; hold out for backtesting."""
    cutoff = pd.Timestamp(holdout_start)
    train = features[features["date"] < cutoff].copy()
    holdout = features[features["date"] >= cutoff].copy()
    return train, holdout


def _build_goals_long_table(features: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each match into two rows (home attack, away attack) for Poisson regression.
    Each row: scoring team, opponent, is_home, goals scored, recency_weight.
    """
    home_rows = pd.DataFrame(
        {
            "team": features["home_team"],
            "opponent": features["away_team"],
            "is_home": 1,
            "goals": features["home_score"],
            "recency_weight": features["recency_weight"],
            "match_date": features["date"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": features["away_team"],
            "opponent": features["home_team"],
            "is_home": 0,
            "goals": features["away_score"],
            "recency_weight": features["recency_weight"],
            "match_date": features["date"],
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def train_outcome_model(train: pd.DataFrame) -> Pipeline:
    """Gradient-boosted multiclass classifier with recency sample weights."""
    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )
    X = train[OUTCOME_FEATURES]
    y = train["outcome"]
    weights = train["recency_weight"]
    model.fit(X, y, sample_weight=weights)
    return model


def train_goals_model(train: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """
    Poisson regression on team/opponent one-hot encodings plus home advantage.
    Learns implicit attack (team) and defense (opponent) coefficients.
    """
    long_df = _build_goals_long_table(train)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "teams",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ["team", "opponent"],
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "poisson",
                PoissonRegressor(alpha=0.01, max_iter=500),
            ),
        ]
    )

    X = long_df[["team", "opponent", "is_home"]]
    y = long_df["goals"]
    weights = long_df["recency_weight"]
    model.fit(X, y, poisson__sample_weight=weights)

    team_names = sorted(
        set(long_df["team"].unique()) | set(long_df["opponent"].unique())
    )
    return model, team_names


def evaluate_on_training(
    outcome_model: Pipeline,
    goals_model: Pipeline,
    train: pd.DataFrame,
) -> None:
    """Print in-sample metrics (sanity check after fitting)."""
    X = train[OUTCOME_FEATURES]
    y = train["outcome"]
    pred_outcome = outcome_model.predict(X)
    acc = accuracy_score(y, pred_outcome)
    print(f"\nOutcome classifier accuracy (training): {acc:.3f}")

    long_df = _build_goals_long_table(train)
    Xg = long_df[["team", "opponent", "is_home"]]
    yg = long_df["goals"]
    pred_goals = goals_model.predict(Xg)
    mae = mean_absolute_error(yg, pred_goals)
    print(f"Goals model MAE (training): {mae:.3f}")


def _wc2022_group_matches(features: pd.DataFrame) -> pd.DataFrame:
    """Filter 2022 World Cup group-stage matches from the feature table."""
    mask = features["date"].between("2022-11-20", "2022-12-02")
    wc = features[mask].copy()
    # Prefer official World Cup fixtures; fall back to all late-Nov 2022 internationals.
    wc_cup = wc[wc["tournament"].str.contains("World Cup", case=False, na=False)]
    return (wc_cup if not wc_cup.empty else wc).sort_values("date")


def backtest_wc2022(
    outcome_model: Pipeline,
    goals_model: Pipeline,
    features: pd.DataFrame,
) -> None:
    """Backtest both models on 2022 World Cup group-stage matches."""
    wc = _wc2022_group_matches(features)
    if wc.empty:
        print("No 2022 World Cup group matches found for backtest.")
        return

    print("\n" + "=" * 72)
    print("2022 FIFA World Cup — Group stage backtest")
    print("=" * 72)

    correct = 0
    goal_errors: list[float] = []

    for _, row in wc.iterrows():
        X_row = row[OUTCOME_FEATURES].to_frame().T
        probs = outcome_model.predict_proba(X_row)[0]
        pred_class = int(np.argmax(probs))
        actual_class = int(row["outcome"])

        home_row = pd.DataFrame(
            [{"team": row["home_team"], "opponent": row["away_team"], "is_home": 1}]
        )
        away_row = pd.DataFrame(
            [{"team": row["away_team"], "opponent": row["home_team"], "is_home": 0}]
        )
        pred_home_goals = float(goals_model.predict(home_row)[0])
        pred_away_goals = float(goals_model.predict(away_row)[0])

        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        goal_errors.extend(
            [
                abs(pred_home_goals - actual_home),
                abs(pred_away_goals - actual_away),
            ]
        )

        if pred_class == actual_class:
            correct += 1

        print(f"\n{row['home_team']} vs {row['away_team']} ({row['date'].date()})")
        print(
            f"  Predicted: {OUTCOME_LABELS[pred_class]} "
            f"({probs[0]:.0%} / {probs[1]:.0%} / {probs[2]:.0%})"
        )
        print(
            f"  Goals pred: {pred_home_goals:.2f} - {pred_away_goals:.2f} | "
            f"Actual: {actual_home} - {actual_away}"
        )
        print(f"  Actual result: {OUTCOME_LABELS[actual_class]}")

    n = len(wc)
    print("\n" + "-" * 72)
    print(f"Outcome accuracy: {correct}/{n} = {correct / n:.1%}")
    print(f"Mean absolute goal error: {np.mean(goal_errors):.3f}")


def main() -> None:
    ensure_dirs()
    features = _load_or_build_features()

    # Train on everything before the 2022 World Cup; backtest on group stage.
    train, _ = _split_train_holdout(features, "2022-11-20")
    print(f"Training on {len(train):,} matches (pre-2022-11-20)")

    outcome_model = train_outcome_model(train)
    goals_model, team_names = train_goals_model(train)

    evaluate_on_training(outcome_model, goals_model, train)
    backtest_wc2022(outcome_model, goals_model, features)

    joblib.dump(outcome_model, OUTCOME_MODEL_PATH)
    joblib.dump(goals_model, GOALS_MODEL_PATH)
    joblib.dump({"teams": team_names}, GOALS_ENCODERS_PATH)
    print(f"\nModels saved to {OUTCOME_MODEL_PATH.parent}")


if __name__ == "__main__":
    main()
