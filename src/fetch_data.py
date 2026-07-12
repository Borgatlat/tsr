"""
Download international match results and FIFA rankings for the last 10 years.

Uses public CSV files from GitHub (no API key). Run:
    python src/fetch_data.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR,
    FIFA_RAW,
    FIFA_URL,
    MATCHES_FILTERED,
    MATCHES_RAW,
    MATCHES_URL,
    ensure_dirs,
    normalize_team,
)


def download_file(url: str, destination: Path) -> None:
    """Stream-download a URL to a local file."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"Saved {destination.name} ({len(response.content):,} bytes)")


def load_matches(use_local: bool = False) -> pd.DataFrame:
    """Load raw match results and keep roughly the last 10 years."""
    ensure_dirs()
    if not use_local or not MATCHES_RAW.exists():
        print(f"Downloading matches from {MATCHES_URL}")
        download_file(MATCHES_URL, MATCHES_RAW)

    df = pd.read_csv(MATCHES_RAW, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)

    cutoff = datetime.now() - timedelta(days=365 * 10)
    recent = df[df["date"] >= cutoff].copy()
    recent = recent.dropna(subset=["home_score", "away_score"])
    recent = recent.sort_values("date").reset_index(drop=True)
    recent.to_csv(MATCHES_FILTERED, index=False)
    print(
        f"Filtered to {len(recent):,} matches from "
        f"{recent['date'].min().date()} to {recent['date'].max().date()}"
    )
    return recent


def load_fifa_rankings(use_local: bool = False) -> pd.DataFrame:
    """Load historical FIFA rankings used for the ranking-gap feature."""
    ensure_dirs()
    if not use_local or not FIFA_RAW.exists():
        print(f"Downloading FIFA rankings from {FIFA_URL}")
        download_file(FIFA_URL, FIFA_RAW)

    rankings = pd.read_csv(FIFA_RAW, parse_dates=["date"])
    rankings["team"] = rankings["team"].map(normalize_team)
    rankings = rankings.dropna(subset=["total_points"])
    # rank 1 = strongest team (most FIFA points on that publication date)
    rankings = rankings.sort_values(["date", "total_points"], ascending=[True, False])
    rankings["rank"] = rankings.groupby("date").cumcount() + 1
    rankings.to_csv(FIFA_RAW, index=False)
    print(f"Loaded {len(rankings):,} FIFA ranking rows")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch international football data")
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Skip download; read existing files from data/",
    )
    args = parser.parse_args()
    load_matches(use_local=args.use_local)
    load_fifa_rankings(use_local=args.use_local)
    print(f"Data ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
