import json

import numpy as np
import pandas as pd
from core.config import CLEAN_DELIVERIES_PATH, CLEAN_MATCHES_PATH, VERSION_DIR
from core.metadata import save_metadata

FEATURES_PATH = VERSION_DIR / "features.parquet"


def build_features():

    print("Loading clean datasets...")
    balls = pd.read_parquet(CLEAN_DELIVERIES_PATH)
    matches = pd.read_parquet(CLEAN_MATCHES_PATH)

    print("Initial shapes:", balls.shape, matches.shape)

    balls = balls.sort_values(["matchId", "inning", "over", "total_balls"]).reset_index(
        drop=True
    )

    print("Fixing no-ball anomalies...")
    mask = balls["isNoBall"] > 1
    balls.loc[mask, "batsman_runs"] += balls.loc[mask, "isNoBall"] - 1
    balls.loc[mask, "isNoBall"] = 1

    print("Expanding wides...")
    balls["repeat"] = np.where(balls["isWide"] > 0, balls["isWide"], 1)
    balls = balls.loc[balls.index.repeat(balls["repeat"])].copy()
    balls.loc[balls["isWide"] > 0, "isWide"] = 1
    balls.drop(columns=["repeat"], inplace=True)

    print("Recomputing legal balls...")
    balls["is_legal"] = ((balls["isWide"] == 0) & (balls["isNoBall"] == 0)).astype(int)

    running_legal_count = balls.groupby(["matchId", "inning", "over"])["is_legal"].cumsum()

    balls["legal_ball"] = (
        running_legal_count.groupby([balls["matchId"], balls["inning"], balls["over"]])
        .shift(1)
        .fillna(0) + 1
    )

    balls = balls[balls["legal_ball"] <= 6].reset_index(drop=True)

    print("Basic match features...")
    balls["legal_ball_1"] = (balls["isWide"] == 0) & (balls["isNoBall"] == 0)

    TOTAL_BALLS = 120

    balls["balls_bowled"] = (
        balls.groupby(["matchId", "inning"])["legal_ball_1"]
        .cumsum()
        .groupby([balls["matchId"], balls["inning"]])
        .shift(fill_value=0)
    )

    balls["balls_remaining"] = TOTAL_BALLS - balls["balls_bowled"]

    balls["over_number"] = balls["over"].astype(int) + 1

    balls["phase_pp"] = (balls["over_number"] <= 6).astype(int)
    balls["phase_middle"] = ((balls["over_number"] > 6) & (balls["over_number"] <= 15)).astype(int)
    balls["phase_death"] = (balls["over_number"] > 15).astype(int)

    print("Score + wickets...")
    balls["total_runs"] = (
        balls["batsman_runs"]
        + balls["isWide"]
        + balls["isNoBall"]
        + balls["Byes"]
        + balls["LegByes"]
        + balls["Penalty"]
    )
    balls["current_score"] = balls.groupby(["matchId", "inning"])["total_runs"].cumsum()

    balls.loc[balls["Penalty"] == 5, "batsman_runs"] = 5
    balls["batsman_runs"] = balls["batsman_runs"] + balls["Byes"] + balls["LegByes"]
    
    balls = balls.reset_index(drop=True)
    balls["is_wicket"] = (balls["player_dismissed"] != "Not Out").astype(int)
    balls["wickets_fallen"] = balls.groupby(["matchId", "inning"])["is_wicket"].cumsum()

    print("Target creation...")
    first_innings_score = (
        balls[balls["inning"] == 0].groupby("matchId")["current_score"].max()
    )

    balls["target"] = balls["matchId"].map(first_innings_score)
    balls.loc[balls["inning"] == 1, "target"] += 1
    balls.loc[balls["inning"] == 0, "target"] = 0

    print("Over-level features...")
    over_runs = (
        balls.groupby(["matchId", "inning", "over_number"])["total_runs"]
        .sum()
        .reset_index(name="over_runs")
    )

    over_runs["last_over_runs"] = over_runs.groupby(["matchId", "inning"])[
        "over_runs"
    ].shift(1)

    balls = balls.merge(
        over_runs[["matchId", "inning", "over_number", "last_over_runs"]],
        on=["matchId", "inning", "over_number"],
        how="left",
    )

    balls["last_over_runs"] = balls["last_over_runs"].fillna(0).astype(int)

    balls["total_balls"] = balls.groupby(["matchId", "inning", "over"]).cumcount() + 1

    print("Applying manual fixes...")
    balls.loc[
        (balls["matchId"] == 1254073)
        & (balls["inning"] == 1)
        & (balls["over"] == 16)
        & (balls["total_balls"] == 5),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [3, 4, 181]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 1254073)
            & (balls["inning"] == 1)
            & (balls["over"] == 16)
            & (balls["total_balls"] > 5)
        ].index
    )

    balls.loc[
        (balls["matchId"] == 1178398)
        & (balls["inning"] == 1)
        & (balls["over"] == 17)
        & (balls["total_balls"] == 5),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [2, 3, 111]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 1178398)
            & (balls["inning"] == 1)
            & (balls["over"] == 17)
            & (balls["total_balls"] > 5)
        ].index
    )

    balls.loc[
        (balls["matchId"] == 729309)
        & (balls["inning"] == 1)
        & (balls["over"] == 18)
        & (balls["total_balls"] == 4),
        ["batsman_runs", "total_runs", "current_score"],
    ] = [6, 6, 131]
    balls = balls.drop(
        balls.loc[
            (balls["matchId"] == 729309)
            & (balls["inning"] == 1)
            & (balls["over"] == 18)
            & (balls["total_balls"] > 4)
        ].index
    )

    balls = balls.sort_values(["matchId", "inning", "over", "total_balls"]).reset_index(
        drop=True
    )

    print("NoBall adjustments...")
    mask = (balls["isNoBall"] == 1) & (balls["player_dismissed"] != "Not Out")
    balls.loc[mask, "isWide"] = 1
    balls.loc[mask, "isNoBall"] = 0

    print("Target Creations...")
    balls["batsman_runs_target"] = balls["batsman_runs"].astype(int)
    balls["isWide_target"] = balls["isWide"].astype(int)
    balls["isNoBall_target"] = balls["isNoBall"].astype(int)
    balls["is_wicket_target"] = balls["is_wicket"].astype(int)

    print("Boundary features...")
    balls["is_boundary"] = balls["batsman_runs"].isin([4, 6]).astype(int)

    def compute_balls_since_boundary(x):
        groups = x.cumsum()
        result = x.groupby(groups).cumcount()
        result[groups == 0] = range(1,(groups == 0).sum()+1)
        return result

    balls["balls_since_boundary"] = balls.groupby(["matchId", "inning"])[
        "is_boundary"
    ].transform(compute_balls_since_boundary)

    balls["balls_since_boundary"] = (
        balls.groupby(["matchId", "inning"])["balls_since_boundary"]
        .shift(1)
        .fillna(0)
    )

    balls['balls_since_boundary'] = balls['balls_since_boundary'].astype(int)

    print("Previous Creation...")
    for col in ["batsman_runs", "isWide", "isNoBall", "is_wicket","total_runs"]:
        balls[f"prev_{col}"] = balls.groupby(["matchId", "inning"])[col].shift(1).fillna(0)

    balls["score_before"] = balls.groupby(["matchId", "inning"])['current_score'].shift(1).fillna(0)
    balls["wickets_before"] = balls.groupby(["matchId", "inning"])['wickets_fallen'].shift(1).fillna(0)

    print("Target progress...")
    balls["percentage_target_achieved"] = np.where(
        balls["inning"] == 0, 0.0, balls["score_before"] / balls["target"]
    )

    balls["percentage_target_achieved"] = (
        balls["percentage_target_achieved"].replace([np.inf, -np.inf], 0).fillna(0)
    )

    print("Merging match metadata...")
    balls = balls.merge(matches[["matchId", "venue"]], on="matchId", how="left")

    print("Run rate features...")
    balls["balls_bowled"] = (
        balls.groupby(["matchId", "inning"])["legal_ball_1"]
        .cumsum()
        .groupby([balls["matchId"], balls["inning"]])
        .shift(fill_value=0)
    )

    balls["balls_remaining"] = TOTAL_BALLS - balls["balls_bowled"]

    balls["overs_bowled"] = balls["balls_bowled"] / 6

    balls["current_run_rate"] = np.where(
        balls["balls_bowled"] > 0, balls["score_before"] / balls["overs_bowled"], 0
    )

    balls["runs_required"] = balls["target"] - balls["score_before"]
    balls["required_run_rate"] = np.where(
        (balls["balls_remaining"] > 0) & (balls["runs_required"] > 0),
        balls["runs_required"] * 6 / balls["balls_remaining"],
        0
    )

    balls.loc[balls["inning"] == 0, "required_run_rate"] = 0
    balls.loc[balls["runs_required"] <= 0, "required_run_rate"] = 0

    print("Adding bowler type feature...")
    with open("../New Data/data/updated_pacers.json", "r") as f:
        pacers = json.load(f)

    balls["is_pacer"] = balls["bowler"].apply(lambda x: 1 if x in pacers else 0)

    balls["over"] = balls["over"] / 20

    season_map = matches.set_index("matchId")["season"]
    balls["season"] = balls["matchId"].map(season_map)

    balls["sin_ball"] = np.sin(2 * np.pi * balls["legal_ball"] / 6)
    balls["cos_ball"] = np.cos(2 * np.pi * balls["legal_ball"] / 6)

    print("Dropping columns...")
    balls.drop(
        columns=[
            'Byes',
            'LegByes',
            'Penalty',
            'ball',
            'balls_bowled',
            'batsman_runs',
            'batting_team',
            'bowling_team',
            'current_score',
            'date',
            'isNoBall',
            'isWide',
            'is_boundary',
            'is_legal',
            'is_wicket',
            'legal_ball',
            'legal_ball_1',
            'over_number',
            'over_number',
            'overs_bowled',
            'player_dismissed',
            'runs_required',
            'total_runs',
            'wickets_fallen'
            ],
        inplace=True,
    )

    print("Normalization...")
    balls["prev_batsman_runs"] /= 6
    balls["prev_total_runs"] /= 6
    balls["balls_remaining"] /= 120
    balls["wickets_before"] /= 10
    balls["balls_since_boundary"] /= 120
    balls["score_before"] /= 200
    balls["target"] /= 200
    balls["last_over_runs"] /= 36
    balls["total_balls"] /= 10

    balls["season"] = ((balls["season"] - 2007)) / 20

    balls["current_run_rate"] /= 36
    balls["required_run_rate"] /= 36
    balls["required_run_rate"] = balls["required_run_rate"].clip(upper=2)

    print("Final shape:", balls.shape)

    print("Saving features...")
    balls.to_parquet(FEATURES_PATH, index=False)

    save_metadata(
        dataset_name="features",
        dataset_path=FEATURES_PATH,
        raw_sources=[str(CLEAN_DELIVERIES_PATH), str(CLEAN_MATCHES_PATH)],
        preprocessing=[
            "ball_level_aggregation",
            "no_ball_correction",
            "wide_ball_expansion",
            "legal_ball_reconstruction",
            "innings_ball_filtering",
            "phase_feature_creation",
            "cumulative_score_tracking",
            "wicket_tracking",
            "target_generation",
            "run_rate_features",
            "required_run_rate_handling",
            "match_metadata_merge",
            "feature_normalization",
            "column_pruning",
        ],
        df=balls,
    )

    print("Features saved successfully.")


if __name__ == "__main__":
    build_features()
