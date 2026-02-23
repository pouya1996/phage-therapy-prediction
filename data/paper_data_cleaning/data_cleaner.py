"""
Paper Data Cleaner & Dataset Builder

This script processes the original paper's training and testing feature files to construct
a balanced, non-overlapping phage-host interaction dataset suitable for ML model training.

WORKFLOW:
    1. Load and analyze original train/test feature files for overlap statistics
    2. Load interaction labels (positive/negative) for both datasets
    3. Rebuild train/test splits with host-based partitioning (no overlap)
    4. Attach per-organism feature vectors (162-d) from the original CSVs
    5. De-normalize features using stored min/max values
    6. Generate an expanded interaction CSV with morphology and concentration columns
    7. Export individual 6x27 feature matrices per phage/host to processed directories

OUTPUT FILES:
    - data/interactions_data.csv              — expanded interaction table
    - data/processed/features/phages/*.csv    — 6x27 phage feature matrices
    - data/processed/features/clinical_isolates/*.csv — 6x27 host feature matrices
    - data/phage_library/phage_metadata.json  — phage name -> morphology mapping

USAGE:
    cd data/paper_data_cleaning
    python data_cleaner.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Add project root so we can import the shared logger / config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = setup_logger(__name__, level=config.logger.get("level"))

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataCleanerConfig:
    """Centralised configuration for the data-cleaning pipeline."""

    # Paths (relative to this script's directory)
    train_features_dir: Path = Path("./data/trainingfeatures/")
    test_features_dir: Path = Path("./data/testfeatures/")
    train_interactions_csv: Path = Path("./data/training_set.csv")
    test_interactions_csv: Path = Path("./data/test_set.csv")
    max_values_csv: Path = Path("./data/max_num.csv")
    min_values_csv: Path = Path("./data/min_num.csv")
    output_interactions_csv: Path = Path("../interactions_data.csv")
    output_phage_features_dir: Path = Path("../../data/processed/features/phages")
    output_host_features_dir: Path = Path("../../data/processed/features/clinical_isolates")
    output_phage_metadata_json: Path = Path("../../data/phage_library/phage_metadata.json")

    # Dataset construction parameters
    hosts_training_ratio: float = 0.70
    training_sample_size: int = 5_676
    total_sample_size: int = 7_500
    random_seed: int = 7
    features_size: int = 324

    # Interaction-expansion parameters
    morphologies: List[str] = field(
        default_factory=lambda: ["Podoviridae", "Myoviridae", "Siphoviridae"]
    )
    concentrations: List[float] = field(
        default_factory=lambda: [0.001] # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    )
    expansion_seed: int = 42

    @property
    def test_sample_size(self) -> int:
        return self.total_sample_size - self.training_sample_size

    @property
    def half_features(self) -> int:
        return self.features_size // 2


# ---------------------------------------------------------------------------
# 1. Analysis helpers
# ---------------------------------------------------------------------------

def _parse_feature_filenames(directory: Path) -> Tuple[List[str], pd.DataFrame]:
    """
    List CSV filenames in *directory* and parse ``Phage,Host`` pairs.

    Returns:
        file_names: raw list of filenames
        df: DataFrame with columns ``['Phage', 'Host']``
    """
    file_names = os.listdir(directory)
    pairs = [name.replace(".csv", "").split(",") for name in file_names]
    df = pd.DataFrame(pairs, columns=["Phage", "Host"])
    return file_names, df


def log_overlap_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log statistics about train/test overlap in the original data."""
    train_phages = set(train_df["Phage"].unique())
    train_hosts = set(train_df["Host"].unique())
    test_phages = set(test_df["Phage"].unique())
    test_hosts = set(test_df["Host"].unique())

    logger.info("=" * 60)
    logger.info("Original dataset statistics")
    logger.info("-" * 60)
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"  Unique phages : {len(train_phages)}")
    logger.info(f"  Unique hosts  : {len(train_hosts)}")
    logger.info(f"Test samples    : {len(test_df)}")
    logger.info(f"  Unique phages : {len(test_phages)}")
    logger.info(f"  Unique hosts  : {len(test_hosts)}")
    logger.info("-" * 60)
    logger.info(f"Overlapping phages: {len(train_phages & test_phages)}")
    logger.info(f"Overlapping hosts : {len(train_hosts & test_hosts)}")
    logger.info("=" * 60)


def log_interaction_counts(
    train_interact: pd.DataFrame,
    test_interact: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log positive / negative interaction counts."""
    logger.info("Interaction label distribution")
    logger.info("-" * 60)
    logger.info(f"Total training interactions : {len(train_interact)}")
    logger.info(f"Total test interactions     : {len(test_interact)}")
    logger.info(
        f"Positive (train / test)    : "
        f"{(train_interact['class'] == 1).sum()} / {(test_interact['class'] == 1).sum()}"
    )
    logger.info(
        f"Negative (train / test)    : "
        f"{(train_interact['class'] == 0).sum()} / {(test_interact['class'] == 0).sum()}"
    )
    logger.info(
        f"Selected negatives (train) : "
        f"{len(train_df) - (train_interact['class'] == 1).sum()}"
    )
    logger.info(
        f"Selected negatives (test)  : "
        f"{len(test_df) - (test_interact['class'] == 1).sum()}"
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 2. Balanced dataset construction
# ---------------------------------------------------------------------------

def build_training_set(
    all_interactions: pd.DataFrame,
    positive_interactions: pd.DataFrame,
    train_hosts: np.ndarray,
    test_hosts: np.ndarray,
    cfg: DataCleanerConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the balanced training set.

    Steps:
        1. Keep all positive interactions whose host is in *train_hosts*.
        2. For phages that only appear in the test-host positives, sample one
           negative interaction with a training host so every phage is covered.
        3. Fill remaining slots with random negative samples from training hosts.

    Returns:
        (train_set, test_positive_interactions)
    """
    train_positives = positive_interactions[
        positive_interactions["host"].isin(train_hosts)
    ]
    test_positives = positive_interactions[
        positive_interactions["host"].isin(test_hosts)
    ]

    # Negative samples for phages missing from training positives
    missing_phage_negatives = all_interactions[
        (all_interactions["phage"].isin(test_positives["phage"]))
        & (all_interactions["host"].isin(train_hosts))
        & (all_interactions["class"] == 0)
    ]
    unique_neg_per_phage = (
        missing_phage_negatives.groupby("phage")
        .apply(lambda g: g.sample(n=1, replace=False, random_state=cfg.random_seed))
        .reset_index(drop=True)
    )

    initial_train = pd.concat([train_positives, unique_neg_per_phage])

    # Top-up with random negatives to hit target size
    remaining = cfg.training_sample_size - len(initial_train)
    eligible = all_interactions[
        (all_interactions["host"].isin(train_hosts))
        & (all_interactions["class"] == 0)
    ]
    additional = eligible.sample(n=remaining, replace=False, random_state=cfg.random_seed)

    train_set = pd.concat([initial_train, additional]).reset_index(drop=True)

    logger.info(
        f"Training set built: {len(train_set)} samples "
        f"(positives={len(train_positives)}, negatives={len(train_set) - len(train_positives)})"
    )
    return train_set, test_positives


def build_test_set(
    all_interactions: pd.DataFrame,
    train_set: pd.DataFrame,
    test_positives: pd.DataFrame,
    test_hosts: np.ndarray,
    cfg: DataCleanerConfig,
) -> pd.DataFrame:
    """
    Construct the test set from remaining interactions.

    Steps:
        1. Exclude rows already used in *train_set*.
        2. Sample negatives from test hosts to reach target size.
    """
    remaining = all_interactions.merge(
        train_set, on=["host", "phage", "class"], how="left", indicator=True
    )
    remaining = remaining.query('_merge == "left_only"').drop("_merge", axis=1)

    needed = cfg.test_sample_size - len(test_positives)
    neg_pool = remaining[
        (remaining["host"].isin(test_hosts)) & (remaining["class"] == 0)
    ]
    additional = neg_pool.sample(n=needed, replace=False, random_state=cfg.random_seed)

    test_set = pd.concat([test_positives, additional]).reset_index(drop=True)

    logger.info(
        f"Test set built    : {len(test_set)} samples "
        f"(positives={len(test_positives)}, negatives={len(test_set) - len(test_positives)})"
    )
    return test_set


# ---------------------------------------------------------------------------
# 3. Feature loading & merging
# ---------------------------------------------------------------------------

def load_features_for_identifiers(
    identifiers: np.ndarray,
    file_names: List[str],
    directory: Path,
    is_phage: bool,
    half: int,
) -> pd.DataFrame:
    """
    Load 162-d feature vectors from the original paper CSVs.

    Args:
        identifiers: unique phage or host accession IDs
        file_names: list of CSV filenames in *directory*
        directory: path to the feature directory
        is_phage: if True slice first half; else second half
        half: number of features per entity (features_size / 2)

    Returns:
        DataFrame indexed by identifier with a single ``features`` column
        containing numpy arrays.
    """
    feature_range = slice(0, half) if is_phage else slice(half, None)
    features_df = pd.DataFrame(index=identifiers, columns=["features"])

    for identifier in identifiers:
        for fname in file_names:
            match = (
                fname.startswith(f"{identifier},")
                if is_phage
                else fname.split(",")[1] == f"{identifier}.csv"
            )
            if match:
                fpath = Path(directory) / fname
                features_df.loc[identifier, "features"] = np.asarray(
                    pd.read_csv(fpath, sep="\t", header=None).iloc[feature_range, 0]
                )
                break

    logger.debug(
        f"Loaded {'phage' if is_phage else 'host'} features for "
        f"{len(identifiers)} identifiers"
    )
    return features_df


def merge_all_features(
    train_file_names: List[str],
    test_file_names: List[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DataCleanerConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge phage / host features from both train and test directories.

    Returns:
        (phages_features, hosts_features) DataFrames with columns
        ``['phage', 'phage_features']`` and ``['host', 'host_features']``
        respectively.
    """
    half = cfg.half_features

    # Phage features
    train_phages = load_features_for_identifiers(
        train_df["Phage"].unique(), train_file_names, cfg.train_features_dir, True, half
    )
    test_phages = load_features_for_identifiers(
        test_df["Phage"].unique(), test_file_names, cfg.test_features_dir, True, half
    )
    phages = pd.concat([train_phages, test_phages]).reset_index()
    phages.columns = ["phage", "phage_features"]

    # Host features
    train_hosts = load_features_for_identifiers(
        train_df["Host"].unique(), train_file_names, cfg.train_features_dir, False, half
    )
    test_hosts = load_features_for_identifiers(
        test_df["Host"].unique(), test_file_names, cfg.test_features_dir, False, half
    )
    hosts = pd.concat([train_hosts, test_hosts]).reset_index()
    hosts.columns = ["host", "host_features"]

    logger.info(
        f"Feature tables ready — phages: {len(phages)}, hosts: {len(hosts)}"
    )
    return phages, hosts


def attach_features(
    interaction_df: pd.DataFrame,
    phages_features: pd.DataFrame,
    hosts_features: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join interaction rows with their phage and host feature vectors."""
    merged = (
        interaction_df
        .merge(phages_features, on="phage", how="inner")
        .merge(hosts_features, on="host", how="inner")
    )
    return merged


# ---------------------------------------------------------------------------
# 4. De-normalisation
# ---------------------------------------------------------------------------

def denormalize_features(
    df: pd.DataFrame,
    cfg: DataCleanerConfig,
) -> pd.DataFrame:
    """
    Reverse the min-max normalisation applied in the original paper.

    The stored max/min CSVs contain 324 values (162 phage + 162 host).
    Formula: ``original = normalised * (max - min) + min``
    """
    max_vals = pd.read_csv(cfg.max_values_csv, header=None).values
    min_vals = pd.read_csv(cfg.min_values_csv, header=None).values

    half = cfg.half_features
    max_phage, max_host = max_vals[:half], max_vals[half:]
    min_phage, min_host = min_vals[:half], min_vals[half:]

    def _denorm(value: np.ndarray, is_phage: bool) -> np.ndarray:
        if is_phage:
            return value * (max_phage - min_phage) + min_phage
        return value * (max_host - min_host) + min_host

    df["phage_features"] = df["phage_features"].apply(lambda x: _denorm(x, True))
    df["host_features"] = df["host_features"].apply(lambda x: _denorm(x, False))

    # Flatten nested array format [[...]] -> [...]
    df["phage_features"] = df["phage_features"].str[0]
    df["host_features"] = df["host_features"].str[0]

    logger.debug(f"De-normalised {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 5. Interaction expansion (morphology x concentration)
# ---------------------------------------------------------------------------

def generate_interaction_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: DataCleanerConfig,
) -> pd.DataFrame:
    """
    Build the expanded interaction table.

    For every base interaction a row is created for each concentration level.
    Each phage is assigned a consistent morphology label.

    Returns:
        DataFrame with columns:
        ``['phage', 'host', 'class', 'dataset', 'morphology', 'concentration']``
    """
    train_copy = train_df.assign(dataset="train")
    test_copy = test_df.assign(dataset="test")
    combined = pd.concat([train_copy, test_copy], ignore_index=True)

    base = combined[["phage", "host", "class", "dataset"]].copy()

    # Consistent morphology mapping
    np.random.seed(cfg.expansion_seed)
    unique_phages = base["phage"].unique()
    morphology_map: Dict[str, str] = {
        p: np.random.choice(cfg.morphologies) for p in unique_phages
    }
    base["morphology"] = base["phage"].map(morphology_map)

    # Replicate across concentrations
    n_conc = len(cfg.concentrations)
    expanded = base.loc[base.index.repeat(n_conc)].reset_index(drop=True)
    expanded["concentration"] = np.tile(cfg.concentrations, len(base))

    logger.info(
        f"Expanded interaction table: {len(expanded):,} rows "
        f"({len(base)} base x {n_conc} concentrations)"
    )
    return expanded


def save_interaction_dataset(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist the interaction DataFrame to CSV and log a summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(df):,} interactions to {output_path}")
    logger.info(f"  Dataset split     : {df['dataset'].value_counts().to_dict()}")
    logger.info(f"  Unique phages     : {df['phage'].nunique()}")
    logger.info(f"  Unique hosts      : {df['host'].nunique()}")
    logger.info(
        f"  Morphology counts : "
        f"{df.groupby('morphology')['phage'].nunique().to_dict()}"
    )
    logger.info(f"  Concentrations    : {sorted(df['concentration'].unique())}")


# ---------------------------------------------------------------------------
# 6. Individual feature export (6x27 matrices)
# ---------------------------------------------------------------------------

def reshape_features_6x27(feature_array: np.ndarray) -> np.ndarray:
    """
    Reshape a flat 162-element feature vector into (6, 27).

    Layout per 27-element block:
        [0:5]   — CHONS counts
        [5]     — molecular weight
        [6:27]  — amino-acid composition (21 values)

    Six blocks correspond to six aggregation statistics
    (mean, max, min, std, var, median).
    """
    if not isinstance(feature_array, np.ndarray):
        feature_array = np.array(feature_array)

    chons, weights, aac = feature_array[:5], [feature_array[5]], feature_array[6:27]

    for idx in range(27, len(feature_array), 27):
        chons = np.concatenate((chons, feature_array[idx : idx + 5]))
        weights.append(feature_array[idx + 5])
        aac = np.concatenate((aac, feature_array[idx + 6 : idx + 27]))

    return np.hstack(
        (
            chons.reshape(6, -1),           # (6, 5)
            np.array(weights).reshape(6, -1),  # (6, 1)
            aac.reshape(6, -1),             # (6, 21)
        )
    )


def export_individual_features(
    train_final: pd.DataFrame,
    test_final: pd.DataFrame,
    cfg: DataCleanerConfig,
) -> None:
    """
    Save per-organism 6x27 feature CSVs and phage metadata JSON.

    Outputs:
        - One CSV per unique phage in ``cfg.output_phage_features_dir``
        - One CSV per unique host in ``cfg.output_host_features_dir``
        - ``cfg.output_phage_metadata_json`` with phage morphology assignments
    """
    phage_dir = cfg.output_phage_features_dir
    host_dir = cfg.output_host_features_dir
    meta_path = cfg.output_phage_metadata_json

    phage_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([train_final, test_final], ignore_index=True)

    # ---- Phages ----
    unique_phages = combined.drop_duplicates(subset=["phage"])[
        ["phage", "phage_features"]
    ].copy()

    np.random.seed(cfg.expansion_seed)
    morph_map = {p: np.random.choice(cfg.morphologies) for p in unique_phages["phage"]}
    unique_phages["morphology"] = unique_phages["phage"].map(morph_map)

    metadata: List[Dict[str, str]] = []
    for _, row in unique_phages.iterrows():
        matrix = reshape_features_6x27(row["phage_features"])
        np.savetxt(phage_dir / f'{row["phage"]}.csv', matrix, delimiter=",", fmt="%.10f")
        metadata.append({"name": row["phage"], "morphology": row["morphology"]})

    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(f"Saved {len(unique_phages)} phage feature files to {phage_dir}/")
    logger.info(f"Saved phage metadata to {meta_path}")

    # ---- Hosts ----
    unique_hosts = combined.drop_duplicates(subset=["host"])[
        ["host", "host_features"]
    ].copy()

    for _, row in unique_hosts.iterrows():
        matrix = reshape_features_6x27(row["host_features"])
        np.savetxt(host_dir / f'{row["host"]}.csv', matrix, delimiter=",", fmt="%.10f")

    logger.info(f"Saved {len(unique_hosts)} host feature files to {host_dir}/")

    # ---- Summary ----
    logger.info(f"Feature dimensions per file: 6 rows x 27 columns")
    morph_dist = pd.Series([m["morphology"] for m in metadata]).value_counts()
    for morph, count in morph_dist.items():
        logger.info(f"  {morph}: {count} phages")


# ---------------------------------------------------------------------------
# 7. Verification helpers
# ---------------------------------------------------------------------------

def verify_morphology_consistency(df: pd.DataFrame) -> None:
    """Assert each phage maps to exactly one morphology."""
    morph_per_phage = df.groupby("phage")["morphology"].nunique()
    if (morph_per_phage == 1).all():
        logger.info("Morphology consistency check passed")
    else:
        inconsistent = morph_per_phage[morph_per_phage > 1]
        logger.warning(f"Inconsistent morphologies found:\n{inconsistent}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Optional[DataCleanerConfig] = None) -> None:
    """Execute the full data-cleaning and export pipeline."""
    if cfg is None:
        cfg = DataCleanerConfig()

    logger.info("=" * 80)
    logger.info("Starting data-cleaning pipeline")
    logger.info("=" * 80)

    # ---- Step 1: Parse original feature filenames ----
    logger.info("Step 1/7 — Parsing original feature filenames")
    train_file_names, train_df = _parse_feature_filenames(cfg.train_features_dir)
    test_file_names, test_df = _parse_feature_filenames(cfg.test_features_dir)
    log_overlap_analysis(train_df, test_df)

    # ---- Step 2: Load interaction labels ----
    logger.info("Step 2/7 — Loading interaction labels")
    train_interact = pd.read_csv(cfg.train_interactions_csv, sep="\t")
    test_interact = pd.read_csv(cfg.test_interactions_csv, sep="\t")
    log_interaction_counts(train_interact, test_interact, train_df, test_df)

    # ---- Step 3: Build balanced train / test splits ----
    logger.info("Step 3/7 — Building balanced train/test splits (host-based)")
    all_interactions = pd.concat([train_interact, test_interact])
    positives = all_interactions[all_interactions["class"] == 1]

    unique_hosts = all_interactions["host"].unique()
    cutoff = int(len(unique_hosts) * cfg.hosts_training_ratio)
    train_hosts = unique_hosts[:cutoff]
    test_hosts = unique_hosts[cutoff:]
    logger.info(
        f"Host split: {len(train_hosts)} train / {len(test_hosts)} test "
        f"({cfg.hosts_training_ratio:.0%} ratio)"
    )

    train_set, test_positives = build_training_set(
        all_interactions, positives, train_hosts, test_hosts, cfg
    )
    test_set = build_test_set(
        all_interactions, train_set, test_positives, test_hosts, cfg
    )

    # ---- Step 4: Attach feature vectors ----
    logger.info("Step 4/7 — Loading and merging feature vectors")
    phages_features, hosts_features = merge_all_features(
        train_file_names, test_file_names, train_df, test_df, cfg
    )
    train_final = attach_features(train_set, phages_features, hosts_features)
    test_final = attach_features(test_set, phages_features, hosts_features)

    # ---- Step 5: De-normalise ----
    logger.info("Step 5/7 — De-normalising features")
    train_final = denormalize_features(train_final, cfg)
    test_final = denormalize_features(test_final, cfg)

    # Shuffle rows
    train_final = train_final.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
    test_final = test_final.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
    logger.info(
        f"Final shapes — train: {train_final.shape}, test: {test_final.shape}"
    )

    # ---- Step 6: Generate expanded interaction CSV ----
    logger.info("Step 6/7 — Generating expanded interaction dataset")
    expanded = generate_interaction_dataset(train_final, test_final, cfg)
    save_interaction_dataset(expanded, cfg.output_interactions_csv)
    verify_morphology_consistency(expanded)

    # ---- Step 7: Export individual feature files ----
    logger.info("Step 7/7 — Exporting individual 6x27 feature files")
    export_individual_features(train_final, test_final, cfg)

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        logger.error(f"Fatal error in data-cleaning pipeline: {exc}", exc_info=True)
        raise
