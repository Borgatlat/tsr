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
    NEUTRAL_OUTCOME_MODEL_PATH,
    OUTCOME_MODEL_PATH,
    ensure_dirs,
)
from features import NEUTRAL_OUTCOME_FEATURES, build_features

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
TEAM_A_OUTCOME_LABELS = {0: "team_a_win", 1: "draw", 2: "team_b_win"}


def _is_neutral_flag(series: pd.Series) -> pd.Series:
    """Normalize neutral column to booleans (CSV may store TRUE/FALSE strings)."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])


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
    Neutral matches use is_home=0 for both teams (no home-advantage boost).
    """
    neutral = _is_neutral_flag(features["neutral"])
    home_is_home = (~neutral).astype(int)

    home_rows = pd.DataFrame(
        {
            "team": features["home_team"],
            "opponent": features["away_team"],
            "is_home": home_is_home,
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


def _swap_neutral_row(row: pd.Series) -> pd.Series:
    """Mirror a neutral feature row so team_a <-> team_b (teaches order invariance)."""
    swapped = row.copy()
    swap_pairs = [
        ("team_a_win_rate_20", "team_b_win_rate_20"),
        ("team_a_avg_goals_scored", "team_b_avg_goals_scored"),
        ("team_a_avg_goals_conceded", "team_b_avg_goals_conceded"),
    ]
    for a_col, b_col in swap_pairs:
        swapped[a_col], swapped[b_col] = row[b_col], row[a_col]

    swapped["win_rate_diff"] = -row["win_rate_diff"]
    swapped["goals_scored_diff"] = -row["goals_scored_diff"]
    swapped["goals_conceded_diff"] = -row["goals_conceded_diff"]
    swapped["fifa_ranking_gap"] = -row["fifa_ranking_gap"]
    swapped["h2h_team_a_win_rate"] = row["h2h_team_b_win_rate"]
    swapped["h2h_team_b_win_rate"] = row["h2h_team_a_win_rate"]
    swapped["abs_fifa_ranking_gap"] = row["abs_fifa_ranking_gap"]

    if row["team_a_outcome"] == 0:
        swapped["team_a_outcome"] = 2
    elif row["team_a_outcome"] == 2:
        swapped["team_a_outcome"] = 0
    return swapped


def build_neutral_training_set(train: pd.DataFrame, augment: bool = True) -> pd.DataFrame:
    """
    Training rows for neutral-venue classifier.
    Uses real neutral matches; optionally mirrors each row with teams swapped.
    """
    neutral = train[_is_neutral_flag(train["neutral"])].copy()
    if neutral.empty:
        return neutral

    if augment:
        mirrored = neutral.apply(_swap_neutral_row, axis=1)
        neutral = pd.concat([neutral, mirrored], ignore_index=True)
    return neutral


def train_outcome_model(train: pd.DataFrame) -> HistGradientBoostingClassifier:
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


def train_neutral_outcome_model(train: pd.DataFrame) -> HistGradientBoostingClassifier | None:
    """Classifier trained only on neutral-venue matches with symmetric features."""
    neutral_train = build_neutral_training_set(train)
    if len(neutral_train) < 50:
        print("Warning: too few neutral matches for dedicated neutral model.")
        return None

    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    X = neutral_train[NEUTRAL_OUTCOME_FEATURES]
    y = neutral_train["team_a_outcome"]
    weights = neutral_train["recency_weight"]
    model.fit(X, y, sample_weight=weights)
    print(f"Neutral outcome model trained on {len(neutral_train):,} rows")
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
    outcome_model: HistGradientBoostingClassifier,
    goals_model: Pipeline,
    train: pd.DataFrame,
    neutral_model: HistGradientBoostingClassifier | None = None,
) -> None:
    """Print in-sample metrics (sanity check after fitting)."""
    X = train[OUTCOME_FEATURES]
    y = train["outcome"]
    pred_outcome = outcome_model.predict(X)
    acc = accuracy_score(y, pred_outcome)
    print(f"\nHome/away outcome classifier accuracy (training): {acc:.3f}")

    if neutral_model is not None:
        neutral_train = build_neutral_training_set(train)
        Xn = neutral_train[NEUTRAL_OUTCOME_FEATURES]
        yn = neutral_train["team_a_outcome"]
        pred_neutral = neutral_model.predict(Xn)
        nacc = accuracy_score(yn, pred_neutral)
        print(f"Neutral outcome classifier accuracy (training): {nacc:.3f}")

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
    outcome_model: HistGradientBoostingClassifier,
    goals_model: Pipeline,
    features: pd.DataFrame,
    neutral_model: HistGradientBoostingClassifier | None = None,
) -> None:
    """Backtest on 2022 World Cup group stage (all neutral venues)."""
    wc = _wc2022_group_matches(features)
    if wc.empty:
        print("No 2022 World Cup group matches found for backtest.")
        return

    use_neutral = neutral_model is not None
    print("\n" + "=" * 72)
    print("2022 FIFA World Cup — Group stage backtest (neutral venues)")
    if use_neutral:
        print("Using dedicated neutral outcome model")
    print("=" * 72)

    correct = 0
    goal_errors: list[float] = []

    for _, row in wc.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        actual_class = int(row["outcome"])
        actual_team_a = int(row.get("team_a_outcome", row["outcome"]))

        if use_neutral:
            X_row = row[NEUTRAL_OUTCOME_FEATURES].to_frame().T
            probs = neutral_model.predict_proba(X_row)[0]
            pred_class = int(np.argmax(probs))
            pred_label = {
                0: f"{home} win",
                1: "Draw",
                2: f"{away} win",
            }[pred_class]
            actual_label = {
                0: f"{home} win",
                1: "Draw",
                2: f"{away} win",
            }[actual_team_a]
            if pred_class == actual_team_a:
                correct += 1
        else:
            X_row = row[OUTCOME_FEATURES].to_frame().T
            probs = outcome_model.predict_proba(X_row)[0]
            pred_class = int(np.argmax(probs))
            pred_label = OUTCOME_LABELS[pred_class]
            actual_label = OUTCOME_LABELS[actual_class]
            if pred_class == actual_class:
                correct += 1

        home_row = pd.DataFrame([{"team": home, "opponent": away, "is_home": 0}])
        away_row = pd.DataFrame([{"team": away, "opponent": home, "is_home": 0}])
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

        print(f"\n{home} vs {away} ({row['date'].date()})")
        print(
            f"  Predicted: {pred_label} "
            f"({probs[0]:.0%} / {probs[1]:.0%} / {probs[2]:.0%})"
        )
        print(
            f"  Goals pred: {pred_home_goals:.2f} - {pred_away_goals:.2f} | "
            f"Actual: {actual_home} - {actual_away}"
        )
        print(f"  Actual result: {actual_label}")

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
    neutral_model = train_neutral_outcome_model(train)
    goals_model, team_names = train_goals_model(train)

    evaluate_on_training(outcome_model, goals_model, train, neutral_model)
    backtest_wc2022(outcome_model, goals_model, features, neutral_model)

    joblib.dump(outcome_model, OUTCOME_MODEL_PATH)
    if neutral_model is not None:
        joblib.dump(neutral_model, NEUTRAL_OUTCOME_MODEL_PATH)
    joblib.dump(goals_model, GOALS_MODEL_PATH)
    joblib.dump({"teams": team_names}, GOALS_ENCODERS_PATH)
    print(f"\nModels saved to {OUTCOME_MODEL_PATH.parent}")


if __name__ == "__main__":
    main()
