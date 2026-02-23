"""
Dataset Preparation Script

This script processes all FASTA files in the raw data directories (clinical_isolates and phages),
extracts protein sequences, and generates feature vectors for machine learning models.
It also auto-syncs phage_metadata.json from interactions_data.csv.

WORKFLOW:
1. Load configuration from config.yaml using config_loader
2. Sync phage_metadata.json with interactions_data.csv (add missing phages)
3. Scan data/raw/clinical_isolates/ and data/raw/phages/ for .fasta files
4. For each FASTA file:
   - Check if features already exist in data/processed/features/
   - If exists, skip processing
   - If not:
     a. Use fasta_parser to extract protein sequences from DNA
     b. Save protein sequences to data/processed/datasets/{filename}_protein.fasta
     c. Use feature_extractor to generate feature vectors
     d. Save features to data/processed/features/{filename}_features.csv

OUTPUT FILES:
- Protein FASTA: data/processed/datasets/{source}/{original_name}.fasta
- Feature CSV: data/processed/features/{source}/{original_name}.csv
  where {source} is either 'clinical_isolates' or 'phages'
- Phage metadata: data/phage_library/phage_metadata.json

USAGE:
    python scripts/prepare_dataset.py
"""

import json
from pathlib import Path
import sys

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger
from src.preprocessing.fasta_parser import process_fasta, save_protein_sequences
from src.preprocessing.feature_extractor import extract_and_aggregate_features, save_features_to_csv

# Initialize logger
logger = setup_logger(__name__, level=config.logger.get("level"))


def sync_phage_metadata() -> None:
    """
    Auto-sync phage_metadata.json with interactions_data.csv.

    Reads unique (phage, morphology) pairs from interactions_data.csv,
    loads the existing phage_metadata.json (if any), and adds any phages
    that are missing. Existing entries are never modified.
    """
    interactions_path = Path(config.paths["interactions"])
    metadata_dir = Path(config.paths["phage_library"])
    metadata_path = metadata_dir / "phage_metadata.json"

    if not interactions_path.exists():
        logger.warning(
            f"Interactions file not found: {interactions_path} — "
            "skipping phage metadata sync"
        )
        return

    # Load interactions
    df = pd.read_csv(interactions_path)
    if "phage" not in df.columns or "morphology" not in df.columns:
        logger.warning(
            "interactions_data.csv is missing 'phage' or 'morphology' column — "
            "skipping phage metadata sync"
        )
        return

    # Unique phage → morphology from interactions (first occurrence wins)
    phage_morph = (
        df[["phage", "morphology"]]
        .drop_duplicates(subset="phage", keep="first")
        .set_index("phage")["morphology"]
        .to_dict()
    )

    # Load existing metadata
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = []

    existing_names = {entry["name"] for entry in metadata}

    # Add missing phages
    added = 0
    for phage, morphology in phage_morph.items():
        if phage not in existing_names:
            metadata.append({"name": phage, "morphology": morphology})
            added += 1

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Phage metadata synced: {len(metadata)} total entries "
        f"({added} added, {len(existing_names)} already existed)"
    )


def prepare_dataset():
    """
    Main function to prepare the dataset by processing all FASTA files.
    
    This function:
    1. Loads configuration
    2. Identifies all FASTA files in raw data directories
    3. Processes each file (protein extraction + feature generation)
    4. Saves results to processed directories
    """
    logger.info("="*80)
    logger.info("Starting dataset preparation")
    logger.info("="*80)
    
    # Configuration is loaded at import time via config_loader
    logger.info("Configuration loaded successfully")
    
    # Step 2: Sync phage_metadata.json from interactions_data.csv
    logger.info("Syncing phage metadata from interactions_data.csv")
    sync_phage_metadata()
    
    # Step 3: Set up paths from config
    data_raw = Path(config.paths['data_raw'])
    data_processed = Path(config.paths['data_processed'])
    
    # Define input and output directories
    input_dirs = [
        data_raw / 'clinical_isolates',
        data_raw / 'phages'
    ]
    
    output_protein_dir = data_processed / 'datasets'
    output_features_dir = data_processed / 'features'
    
    # Create output directories if they don't exist
    output_protein_dir.mkdir(parents=True, exist_ok=True)
    output_features_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directories: {[str(d) for d in input_dirs]}")
    logger.info(f"Output protein directory: {output_protein_dir}")
    logger.info(f"Output features directory: {output_features_dir}")
    
    # Step 3: Collect all FASTA files with source directory tracking
    fasta_files = []
    for input_dir in input_dirs:
        if input_dir.exists():
            files = list(input_dir.glob('*.fasta')) + list(input_dir.glob('*.fa')) + list(input_dir.glob('*.fna'))
            # Store tuple of (file_path, source_name)
            source_name = input_dir.name  # 'clinical_isolates' or 'phages'
            fasta_files.extend([(f, source_name) for f in files])
            logger.info(f"Found {len(files)} FASTA file(s) in {input_dir}")
        else:
            logger.warning(f"Directory does not exist: {input_dir}")
    
    if not fasta_files:
        logger.warning("No FASTA files found to process")
        return
    
    logger.info(f"Total FASTA files to process: {len(fasta_files)}")
    logger.info("-"*80)
    
    # Step 4: Process each FASTA file
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for fasta_file, source_name in fasta_files:
        try:
            # Create source-specific subdirectories
            source_protein_dir = output_protein_dir / source_name
            source_features_dir = output_features_dir / source_name
            source_protein_dir.mkdir(parents=True, exist_ok=True)
            source_features_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if features already exist
            expected_feature_file = source_features_dir / f"{fasta_file.stem}.csv"
            
            if expected_feature_file.exists():
                logger.info(f"SKIP: {fasta_file.name} (features already exist)")
                skipped_count += 1
                continue
            
            logger.info(f"PROCESSING: {fasta_file.name} [{source_name}]")
            
            # Step 4a: Extract protein sequences using fasta_parser
            protein_sequences = process_fasta(fasta_file)
            
            if not protein_sequences:
                logger.warning(f"  - No protein sequences extracted from {fasta_file.name}, skipping")
                error_count += 1
                continue
            
            # Step 4b: Save protein sequences to source-specific directory
            protein_output_filename = f"{fasta_file.stem}.fasta"
            protein_output_file = save_protein_sequences(
                protein_sequences, 
                source_protein_dir,
                protein_output_filename
            )
            
            # Step 4c: Extract features from protein sequences (not FASTA path)
            features = extract_and_aggregate_features(protein_sequences, source_name=fasta_file.name)
            
            # Step 4d: Save features to CSV in source-specific directory
            features_output_filename = f"{fasta_file.stem}.csv"
            features_output_file = save_features_to_csv(
                features,
                source_features_dir,
                features_output_filename
            )
            
            logger.info(f"SUCCESS: {fasta_file.name} -> Proteins: {source_name}/{protein_output_file.name}, Features: {source_name}/{features_output_file.name}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"ERROR processing {fasta_file.name}: {e}")
            error_count += 1
            continue
    
    # Step 5: Summary
    logger.info("="*80)
    logger.info("Dataset preparation completed")
    logger.info(f"Total files found: {len(fasta_files)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("="*80)


if __name__ == '__main__':
    try:
        prepare_dataset()
    except Exception as e:
        logger.error(f"Fatal error in dataset preparation: {e}")
        raise
