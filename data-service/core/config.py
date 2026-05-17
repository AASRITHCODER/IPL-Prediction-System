from pathlib import Path

DATASET_VERSION = "v4_gamma"

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_MATCHES = BASE_DIR / "../New Data/matches_updated_ipl_upto_2025.csv"
RAW_DELIVERIES = BASE_DIR / "../New Data/deliveries_updated_ipl_upto_2025.csv"

PROCESSED_DIR = BASE_DIR / "../ml-service/data/processed"

VERSION_DIR = PROCESSED_DIR / DATASET_VERSION

CLEAN_MATCHES_PATH = VERSION_DIR / "clean_matches.parquet"
CLEAN_DELIVERIES_PATH = VERSION_DIR / "clean_deliveries.parquet"

METADATA_PATH = BASE_DIR / f"../ml-service/data/metadata/{DATASET_VERSION}"

NON_STRIKER_RATE = 542 / 13581  # non_striker_outs/total_wickets
