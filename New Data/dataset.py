import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IPLDataset(Dataset):
    """
    Sequence Dataset for IPL Run Prediction LSTM.

    Input:
        Last `sequence_length` legal deliveries

    Output:
        Current ball runs (0-6)

    Notes:
    - Prevents sequence crossing between matches/innings
    - Uses left zero-padding for early balls
    - Separates input features and targets
    - Supports future embedding integration
    - Keeps categorical IDs separate
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_year: int,
        sequence_length: int = 30,
    ):
        self.sequence_length = sequence_length
        self.train_year = train_year

        # Chronological filtering
        df = df[df["season"] < train_year].copy()

        # -------------------------------------------------
        # 1. SORT DATA PROPERLY
        # -------------------------------------------------
        df = df.sort_values(
            by=["matchId", "inning", "over", "total_balls"]
        ).reset_index(drop=True)

        # -------------------------------------------------
        # 2. ENCODE CATEGORICAL FEATURES
        # -------------------------------------------------
        # IMPORTANT:
        # TabTransformer still requires integer categorical IDs.
        # DO NOT one-hot encode.

        # -------------------------------------------------
        # LOAD STABLE GLOBAL MAPPINGS
        # -------------------------------------------------

        with open("../New Data/data/all_players.json", "r") as f:
            all_players = json.load(f)

        with open("../New Data/data/all_venues.json", "r") as f:
            all_venues = json.load(f)

        # 0 RESERVED FOR PADDING
        self.player2idx = {player: idx + 1 for idx, player in enumerate(all_players)}

        self.venue2idx = {venue: idx + 1 for idx, venue in enumerate(all_venues)}

        # -------------------------------------------------
        # APPLY ENCODING
        # -------------------------------------------------

        df["batter_id"] = df["batsman"].map(self.player2idx)
        df["non_striker_id"] = df["non_striker"].map(self.player2idx)
        df["bowler_id"] = df["bowler"].map(self.player2idx)
        df["venue_id"] = df["venue"].map(self.venue2idx)

        # df = df.dropna().reset_index(drop=True)

        # -------------------------------------------------
        # 3. DEFINE INPUT FEATURES
        # -------------------------------------------------
        # ONLY numerical/context features here.
        # IDs kept separately for future embeddings.

        self.feature_columns = [
            "inning",
            "over",
            "total_balls",
            "balls_remaining",
            "phase_pp",
            "phase_middle",
            "phase_death",
            "target",
            "last_over_runs",
            "balls_since_boundary",
            "prev_batsman_runs",
            "prev_isWide",
            "prev_isNoBall",
            "prev_is_wicket",
            "prev_total_runs",
            "score_before",
            "wickets_before",
            "percentage_target_achieved",
            "current_run_rate",
            "required_run_rate",
            "is_pacer",
            "season",
            "sin_ball",
            "cos_ball",
        ]

        # -------------------------------------------------
        # 4. STORE CATEGORICAL IDS SEPARATELY
        # -------------------------------------------------
        self.categorical_columns = [
            "batter_id",
            "non_striker_id",
            "bowler_id",
            "venue_id",
        ]

        # -------------------------------------------------
        # FEATURE & EMBEDDING METADATA
        # -------------------------------------------------

        self.numerical_dim = len(self.feature_columns)
        self.categorical_dim = len(self.categorical_columns)

        self.num_players = len(self.player2idx) + 1
        self.num_venues = len(self.venue2idx) + 1

        # -------------------------------------------------
        # 5. BUILD SEQUENCES
        # -------------------------------------------------
        self.X_numerical = []
        self.X_categorical = []
        self.y_runs = []
        self.y_wide = []
        self.y_noball = []
        self.y_wicket = []

        grouped = df.groupby(["matchId", "inning"])

        for (_, _), group in grouped:
            group = group.reset_index(drop=True)

            numerical_features = group[self.feature_columns].values
            categorical_features = group[self.categorical_columns].values
            runs_targets = group["batsman_runs_target"].values
            wide_targets = group["isWide_target"].values
            noball_targets = group["isNoBall_target"].values
            wicket_targets = group["is_wicket_target"].values

            for idx in range(len(group)):

                start_idx = max(0, idx - self.sequence_length + 1)

                numerical_seq = numerical_features[start_idx : idx + 1]
                categorical_seq = categorical_features[start_idx : idx + 1]

                # ---------------------------------------------
                # LEFT ZERO PADDING
                # ---------------------------------------------
                pad_size = self.sequence_length - len(numerical_seq)

                if pad_size > 0:
                    numerical_padding = np.zeros(
                        (pad_size, len(self.feature_columns)),
                        dtype=np.float32,
                    )

                    categorical_padding = np.zeros(
                        (pad_size, len(self.categorical_columns)),
                        dtype=np.int64,
                    )

                    numerical_seq = np.vstack([numerical_padding, numerical_seq])

                    categorical_seq = np.vstack([categorical_padding, categorical_seq])

                self.X_numerical.append(numerical_seq)
                self.X_categorical.append(categorical_seq)
                self.y_runs.append(runs_targets[idx])
                self.y_wide.append(wide_targets[idx])
                self.y_noball.append(noball_targets[idx])
                self.y_wicket.append(wicket_targets[idx])

        # -------------------------------------------------
        # 6. CONVERT TO NUMPY
        # -------------------------------------------------
        self.X_numerical = np.array(self.X_numerical, dtype=np.float32)

        self.X_categorical = np.array(self.X_categorical, dtype=np.int64)

        # Multi-class classification
        self.y_runs = np.array(self.y_runs, dtype=np.int64)

        # Binary classification
        self.y_wide = np.array(self.y_wide, dtype=np.float32)
        self.y_noball = np.array(self.y_noball, dtype=np.float32)
        self.y_wicket = np.array(self.y_wicket, dtype=np.float32)

        print("Dataset Built Successfully")
        print(f"Samples: {len(self.y_runs)}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Numerical Shape: {self.X_numerical.shape}")
        print(f"Categorical Shape: {self.X_categorical.shape}")

    def __len__(self):
        return len(self.y_runs)

    def __getitem__(self, idx):

        numerical_tensor = torch.tensor(
            self.X_numerical[idx],
            dtype=torch.float32,
        )

        categorical_tensor = torch.tensor(
            self.X_categorical[idx],
            dtype=torch.long,
        )

        runs_target = torch.tensor(
            self.y_runs[idx],
            dtype=torch.long,
        )

        wide_target = torch.tensor(
            self.y_wide[idx],
            dtype=torch.float32,
        )

        noball_target = torch.tensor(
            self.y_noball[idx],
            dtype=torch.float32,
        )

        wicket_target = torch.tensor(
            self.y_wicket[idx],
            dtype=torch.float32,
        )

        return {
            "numerical_features": numerical_tensor,
            "categorical_features": categorical_tensor,
            "runs_target": runs_target,
            "wide_target": wide_target,
            "noball_target": noball_target,
            "wicket_target": wicket_target,
        }


if __name__ == "__main__":

    df = pd.read_parquet("../ml-service/data/processed/v4_gamma/features.parquet")

    train_year = 2010
    train_year = (train_year - 2007) / 20

    dataset = IPLDataset(df=df, train_year=train_year, sequence_length=30)

    sample = dataset[0]

    print("\nSample Shapes")
    print(sample["numerical_features"].shape)
    print(sample["categorical_features"].shape)
    print(sample["runs_target"])
