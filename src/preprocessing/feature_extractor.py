"""
Feature Engineering Module for Protein Sequence Analysis

This module extracts comprehensive features from protein sequences to support
machine learning models for protein-protein interaction prediction.

FEATURES EXTRACTED:
1. Amino Acid Composition (AAC): 21 features
   - Fraction of each of the 20 standard amino acids in the sequence
   - Plus fraction of unknown/ambiguous residues ('*')

2. Physical-Chemical Properties: 5 features
   - Total atomic composition: Carbon (C), Hydrogen (H), Oxygen (O), Nitrogen (N), Sulfur (S)
   - Calculated by summing elemental composition across all amino acids

3. Molecular Weight: 1 feature
   - Monoisotopic molecular weight of the protein sequence

AGGREGATION METHOD:
For each organism, the module computes 6 statistical measures across all proteins:
- Mean, Maximum, Minimum, Standard Deviation, Variance, Median

OUTPUT:
- CSV files with 6 rows × 27 columns (21 AAC + 5 CHONS + 1 MW)
- Each row represents a statistical measure
- Each column represents a feature dimension

WORKFLOW:
1. Receive protein sequences as dictionary input (from fasta_parser or other sources)
2. Extract features (AAC, CHONS, MW) from each protein sequence
3. Aggregate features using statistical measures (mean, max, min, std, var, median)
4. Return aggregated feature matrix as numpy array

USAGE:
    Basic usage:
    
    >>> from pathlib import Path
    >>> from src.preprocessing.fasta_parser import process_fasta
    >>> from src.preprocessing.feature_extractor import extract_and_aggregate_features, save_features_to_csv
    >>> 
    >>> # First, extract proteins from DNA FASTA
    >>> fasta_file = Path('data/raw/phages/phage1.fasta')
    >>> proteins = process_fasta(fasta_file)
    >>> 
    >>> # Then extract features from protein sequences
    >>> features = extract_and_aggregate_features(proteins)
    >>> print(features.shape)  # (6, 27)
    >>> 
    >>> # Save to CSV
    >>> output_dir = Path('data/processed/features')
    >>> output_file = save_features_to_csv(features, output_dir, 'phage1_features.csv')
    >>> print(f"Saved to: {output_file}")
    
    Working with protein FASTA files directly:
    
    >>> from Bio import SeqIO
    >>> 
    >>> # Read protein sequences from FASTA
    >>> protein_file = Path('data/processed/datasets/organism_protein.fasta')
    >>> proteins = {record.id: str(record.seq) for record in SeqIO.parse(protein_file, 'fasta')}
    >>> 
    >>> # Extract and save features
    >>> features = extract_and_aggregate_features(proteins)
    >>> output_dir = Path('data/processed/features')
    >>> output_file = save_features_to_csv(features, output_dir, 'organism_features.csv')
    
    Process multiple files:
    
    >>> from src.preprocessing.fasta_parser import process_fasta
    >>> 
    >>> input_dir = Path('data/raw/phages')
    >>> output_dir = Path('data/processed/features')
    >>> 
    >>> for fasta_file in input_dir.glob('*.fasta'):
    ...     proteins = process_fasta(fasta_file)  # DNA -> proteins
    ...     features = extract_and_aggregate_features(proteins)
    ...     output_filename = f"{fasta_file.stem}_features.csv"
    ...     output_file = save_features_to_csv(features, output_dir, output_filename)
    ...     print(f"Processed: {fasta_file.name} -> {output_file.name}")

EXAMPLE OUTPUT:
    The output CSV file contains 6 rows (statistics) × 27 columns (features):
    
    Row 0: Mean values across all proteins
    Row 1: Maximum values
    Row 2: Minimum values
    Row 3: Standard deviation
    Row 4: Variance
    Row 5: Median values
    
    Columns 0-20: AAC for A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,*
    Columns 21-25: C, H, O, N, S counts
    Column 26: Molecular weight
"""

import pandas as pd
import numpy as np
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Dict
from pathlib import Path
import warnings
from ..utils.logger_utils import setup_logger
from ..utils.config_loader import config

# Initialize logger
logger = setup_logger(__name__)

# Suppress noisy Biopython deprecation / sequence warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="Bio")


def calculate_aac(sequence: str) -> list:
    """
    Calculate the amino acid composition (AAC) for a protein sequence.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        List of 21 float values representing the fraction of each amino acid:
        - 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY)
        - 1 unknown/ambiguous residue ('*')
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY*'
    seq_length = len(sequence)
    if seq_length == 0:
        return [0.0] * 21
    aa_counts = Counter(sequence)
    # Compute fraction of each amino acid in the sequence
    return [aa_counts.get(aa, 0) / seq_length for aa in amino_acids]


def calculate_physical_chemical_features(sequence: str) -> list:
    """
    Calculate the total atomic composition of C, H, O, N, S for a protein sequence.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        List of 5 integer values representing total count of each element:
        - Carbon (C)
        - Hydrogen (H)
        - Oxygen (O)
        - Nitrogen (N)
        - Sulfur (S)
        
    Note:
        Ambiguous residues (X, U, B, Z) are removed before counting.
        Each amino acid's elemental composition is based on its molecular formula.
    """
    # Remove any ambiguous residues before counting
    clean_seq = sequence.replace('X', '').replace('U', '').replace('B', '').replace('Z', '')
    elements = 'CHONS'
    # Map each amino acid to its elemental composition
    amino_acid_elements = {
        'A': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 0},
        'C': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
        'D': {'C': 4, 'H': 7, 'O': 4, 'N': 1, 'S': 0},
        'E': {'C': 5, 'H': 9, 'O': 4, 'N': 1, 'S': 0},
        'F': {'C': 9, 'H': 11, 'O': 2, 'N': 1, 'S': 0},
        'G': {'C': 2, 'H': 5, 'O': 2, 'N': 1, 'S': 0},
        'H': {'C': 6, 'H': 9, 'O': 2, 'N': 3, 'S': 0},
        'I': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
        'K': {'C': 6, 'H': 14, 'O': 2, 'N': 2, 'S': 0},
        'L': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
        'M': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 1},
        'N': {'C': 4, 'H': 8, 'O': 3, 'N': 2, 'S': 0},
        'P': {'C': 5, 'H': 9, 'O': 2, 'N': 1, 'S': 0},
        'Q': {'C': 5, 'H': 10, 'O': 3, 'N': 2, 'S': 0},
        'R': {'C': 6, 'H': 14, 'O': 2, 'N': 4, 'S': 0},
        'S': {'C': 3, 'H': 7, 'O': 3, 'N': 1, 'S': 0},
        'T': {'C': 4, 'H': 9, 'O': 3, 'N': 1, 'S': 0},
        'V': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 0},
        'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2, 'S': 0},
        'Y': {'C': 9, 'H': 11, 'O': 3, 'N': 1, 'S': 0}
    }
    aa_counts = Counter(clean_seq)
    # Sum per element across all amino acids in the sequence
    return [
        sum(amino_acid_elements.get(aa, {}).get(el, 0) * count for aa, count in aa_counts.items())
        for el in elements
    ]


def calculate_molecular_weight(sequence: str) -> float:
    """
    Calculate the monoisotopic molecular weight of a protein sequence.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        Float value representing the monoisotopic molecular weight in Daltons (Da)
        
    Note:
        Uses Biopython's ProteinAnalysis with monoisotopic mass calculation.
        Ambiguous residues (X, U, B, Z) are removed before calculation.
    """
    clean_seq = sequence.replace('X', '').replace('U', '').replace('B', '').replace('Z', '')
    if len(clean_seq) == 0:
        return 0.0
    analysed_seq = ProteinAnalysis(clean_seq)
    analysed_seq.monoisotopic = True  # Use monoisotopic masses
    return analysed_seq.molecular_weight()


def extract_features_from_proteins(protein_sequences: Dict[str, str]) -> np.ndarray:
    """
    Extract features from a dictionary of protein sequences.
    
    Args:
        protein_sequences: Dictionary with protein IDs as keys and protein sequences as values
                          (output from fasta_parser.process_fasta)
                          
    Returns:
        numpy.ndarray of shape (num_proteins, 27) containing feature vectors for each protein:
        - Columns 0-20: Amino acid composition (AAC)
        - Columns 21-25: Physical-chemical properties (C, H, O, N, S counts)
        - Column 26: Molecular weight
        
    Example:
        >>> proteins = {'>protein_1': 'MPKKLM', '>protein_2': 'MWRNA'}
        >>> features = extract_features_from_proteins(proteins)
        >>> features.shape
        (2, 27)
    """
    features = []
    
    for protein_id, sequence in protein_sequences.items():
        aac = calculate_aac(sequence)
        chons = calculate_physical_chemical_features(sequence)
        mw = [calculate_molecular_weight(sequence)]
        features.append(aac + chons + mw)
    
    logger.debug(f"Extracted features from {len(features)} protein(s)")
    return np.array(features)


def aggregate_statistics(features_array: np.ndarray) -> np.ndarray:
    """
    Compute statistical measures across all protein feature vectors.
    
    Args:
        features_array: numpy.ndarray of shape (num_proteins, 27) containing feature vectors
        
    Returns:
        numpy.ndarray of shape (6, 27) containing aggregated statistics:
        - Row 0: Mean of each feature across all proteins
        - Row 1: Maximum value of each feature
        - Row 2: Minimum value of each feature
        - Row 3: Standard deviation of each feature
        - Row 4: Variance of each feature
        - Row 5: Median of each feature
        
    Example:
        >>> features = np.array([[1, 2, 3], [4, 5, 6]])
        >>> stats = aggregate_statistics(features)
        >>> stats.shape
        (6, 3)
    """
    aggregated = np.vstack([
        features_array.mean(axis=0),
        features_array.max(axis=0),
        features_array.min(axis=0),
        features_array.std(axis=0),
        features_array.var(axis=0),
        np.median(features_array, axis=0)
    ])
    return aggregated


def extract_and_aggregate_features(protein_sequences: Dict[str, str], source_name: str = "proteins") -> np.ndarray:
    """
    Main function to extract and aggregate features from protein sequences.
    
    This function:
    1. Receives protein sequences as dictionary input
    2. Extracts features (AAC, CHONS, MW) from each protein
    3. Aggregates features using statistical measures
    
    Args:
        protein_sequences: Dictionary with protein IDs as keys and protein sequences as values
                          (e.g., output from fasta_parser.process_fasta or from reading protein FASTA)
        source_name: Name to use in log messages for identifying the source (default: "proteins")
        
    Returns:
        numpy.ndarray of shape (6, 27) containing aggregated feature statistics:
        - 6 rows: mean, max, min, std, var, median
        - 27 columns: 21 AAC + 5 CHONS + 1 MW
        
    Example:
        >>> from src.preprocessing.fasta_parser import process_fasta
        >>> from pathlib import Path
        >>> 
        >>> # From DNA FASTA
        >>> fasta_file = Path('data/raw/phages/organism.fasta')
        >>> proteins = process_fasta(fasta_file)
        >>> features = extract_and_aggregate_features(proteins, source_name="organism.fasta")
        >>> features.shape
        (6, 27)
        >>> 
        >>> # From protein FASTA
        >>> from Bio import SeqIO
        >>> protein_file = Path('data/processed/datasets/organism_protein.fasta')
        >>> proteins = {record.id: str(record.seq) for record in SeqIO.parse(protein_file, 'fasta')}
        >>> features = extract_and_aggregate_features(proteins, source_name="organism_protein.fasta")
    """
    if not protein_sequences:
        logger.error(f"No protein sequences provided from {source_name}")
        raise ValueError(f"No protein sequences provided from {source_name}")
    
    # Extract features from each protein
    features_array = extract_features_from_proteins(protein_sequences)
    
    # Aggregate features using statistical measures
    aggregated_features = aggregate_statistics(features_array)
    
    logger.info(f"Generated features (6×27) from {source_name}")
    return aggregated_features


def save_features_to_csv(features: np.ndarray, output_dir: Path, output_filename: str) -> Path:
    """
    Save aggregated features to a CSV file.
    
    Args:
        features: numpy.ndarray of shape (6, 27) containing aggregated features
        output_dir: Path object for output directory (required)
        output_filename: Name for the output file (e.g., 'organism_features.csv')
        
    Returns:
        Path object to the saved CSV file
        
    Example:
        >>> from pathlib import Path
        >>> fasta_file = Path('data/raw/phages/organism.fasta')
        >>> features = extract_and_aggregate_features(fasta_file)
        >>> output_dir = Path('data/processed/features')
        >>> output_file = save_features_to_csv(features, output_dir, 'organism_features.csv')
        >>> # Saves to: data/processed/features/organism_features.csv
    """
    # Convert to Path object
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create full output path
    output_path = output_dir / output_filename
    
    pd.DataFrame(features).to_csv(output_path, index=False, header=False)
    logger.info(f"Saved features to {output_filename}")
    
    return output_path