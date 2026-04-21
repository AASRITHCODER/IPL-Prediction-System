import json
from pathlib import Path

import pandas as pd
from core.config import CLEAN_DELIVERIES_PATH, CLEAN_MATCHES_PATH, RAW_DELIVERIES
from core.metadata import save_metadata


def clean_deliveries():

    print("Loading deliveries dataset...")
    balls = pd.read_csv(RAW_DELIVERIES)

    print("Converting date...")
    balls["date"] = pd.to_datetime(balls["date"])

    print("Filling missing values...")
    balls["isWide"] = balls["isWide"].fillna(0)
    balls["isNoBall"] = balls["isNoBall"].fillna(0)
    balls["player_dismissed"] = balls["player_dismissed"].fillna("Not Out")
    balls["Byes"] = balls["Byes"].fillna(0)
    balls["LegByes"] = balls["LegByes"].fillna(0)
    balls["Penalty"] = balls["Penalty"].fillna(0)

    print("Creating Total runs...")
    balls["total_runs"] = (
        balls["batsman_runs"]
        + balls["isWide"]
        + balls["isNoBall"]
        + balls["Byes"]
        + balls["LegByes"]
        + balls["Penalty"]
    )

    print("Dropping unnecessary columns...")
    balls.drop(
        ["over_ball", "dismissal_kind", "extras"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    print("Keeping only innings 1 and 2...")
    balls = balls[balls["inning"].isin([1, 2])].copy()

    print("Mapping innings...")
    balls["inning"] = balls["inning"].map({1: 0, 2: 1})

    print("Converting types...")
    balls["batsman_runs"] = balls["batsman_runs"].astype(int)
    balls["isWide"] = balls["isWide"].astype(int)
    balls["isNoBall"] = balls["isNoBall"].astype(int)

    print("Removing washed-out matches from deliveries...")
    matches = pd.read_parquet(CLEAN_MATCHES_PATH)
    valid_match_ids = set(matches["matchId"].unique())

    balls = balls[balls["matchId"].isin(valid_match_ids)]

    with open("../New Data/data/updated_players.json", "r") as file:
        players = json.load(file)

    print("Mapping Players...")
    balls["batsman"] = balls["batsman"].map(players)
    balls["non_striker"] = balls["non_striker"].map(players)
    balls["bowler"] = balls["bowler"].map(players)
    balls["player_dismissed"] = balls["player_dismissed"].map(players)

    print("Generating total_balls and Sorting by date...")
    balls["total_balls"] = balls.groupby(["matchId", "inning", "over"]).cumcount() + 1
    balls = balls.sort_values(
        ["date", "matchId", "inning", "over", "total_balls"]
    ).reset_index(drop=True)

    print("Removing duplicate rows...")
    balls = balls.drop_duplicates()

    print("Saving parquet...")
    balls.to_parquet(CLEAN_DELIVERIES_PATH, index=False)

    print("Clean deliveries saved successfully.")

    save_metadata(
        dataset_name="clean_deliveries",
        dataset_path=CLEAN_DELIVERIES_PATH,
        raw_sources=[str(RAW_DELIVERIES), str(CLEAN_MATCHES_PATH)],
        preprocessing=[
            "date_type_conversion",
            "extras_null_imputation",
            "dismissal_null_imputation",
            "total_runs_computation",
            "column_pruning",
            "inning_filtering_and_mapping",
            "type_casting_numeric_fields",
            "invalid_match_removal_using_clean_matches",
            "duplicate_removal",
            "chronological_sorting",
        ],
        df=balls,
    )


if __name__ == "__main__":
    clean_deliveries()
