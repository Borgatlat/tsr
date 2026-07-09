# World Cup Match Prediction Model

Predict international football match outcomes and the number of goals each team scores, using roughly the last 10 years of international match data. Recent seasons are weighted more heavily via exponential decay so the model reflects current team form.

## Goal

- **Outcome model**: win / draw / loss probabilities for the home team
- **Goals model**: expected goals for each team (Poisson regression with per-team attack/defense strength)
- **Recency weighting**: `weight = 0.85 ** years_ago` (a match from this year counts about 3× more than one from 5 years ago, since `0.85^5 ≈ 0.44`)

## Project structure

```
data/           # Downloaded match and ranking CSVs (gitignored)
src/            # Python source code
notebooks/      # Exploratory analysis
models/         # Trained model artifacts (gitignored)
predict.py      # CLI entry point at project root
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data sources (no API key required)

Match results are downloaded automatically from the public [martj42/international_results](https://github.com/martj42/international_results) GitHub repository (`results.csv`).

FIFA rankings come from [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) (`ranking_fifa_historical.csv`).

### Manual download (if automatic fetch fails)

1. **Matches**: save [results.csv](https://raw.githubusercontent.com/martj42/international_results/master/results.csv) to `data/results.csv`
2. **FIFA rankings**: save [ranking_fifa_historical.csv](https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv) to `data/fifa_rankings.csv`

Then run:

```bash
python src/fetch_data.py --use-local
```

## Usage

```bash
# 1. Download and filter the last 10 years of data
python src/fetch_data.py

# 2. Build features (prints a sample with recency weights)
python src/features.py

# 3. Train models, backtest 2022 World Cup group stage, save artifacts
python src/train_model.py

# 4. Predict a matchup
python predict.py Spain Belgium
```

## Features (per match)

| Feature | Description |
|---------|-------------|
| Home/away win rate | Win rate over each team's last 20 games before the match |
| Avg goals scored / conceded | Rolling averages from prior matches |
| Head-to-head | Home team's win rate in prior meetings |
| FIFA ranking gap | Home rank minus away rank (lower rank number = stronger) |
| Recency weight | `0.85 ** years_ago` applied as `sample_weight` during training |

## Models

1. **Outcome classifier** — `HistGradientBoostingClassifier` (multiclass: home win / draw / away win)
2. **Goals regressor** — `PoissonRegressor` on team-level rows (home and away as separate observations) with one-hot team and opponent encodings

Both models use `sample_weight` from the recency column.
