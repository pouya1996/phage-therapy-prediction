"""
FASTA to Protein Sequence Converter

This module processes FASTA format DNA sequences and identifies Open Reading Frames (ORFs),
translating them into protein sequences.

Functions:
    process_fasta: Process single FASTA file and extract protein sequences from ORFs
    save_protein_sequences: Save processed results to FASTA file
"""
from pathlib import Path
from typing import Dict, List, Tuple
from ..utils.logger_utils import setup_logger
from ..utils.config_loader import config

# Initialize logger
logger = setup_logger(__name__, level=config.logger.get("level"))

def read_fasta(fasta_path: Path) -> Dict[str, str]:
    """
    Reads a FASTA file and returns a dictionary of sequences.
    
    Args:
        fasta_path: Path object to the FASTA file
        
    Returns:
        Dictionary with accession numbers as keys and sequences as values
    """
    sequences = {}
    
    try:
        with open(fasta_path, 'r') as f:
            accession = ''
            seq = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save the previous accession and sequence
                    if accession and seq:
                        sequences[accession] = seq
                    accession = line[1:]  # Remove '>' from header
                    seq = ''
                else:
                    # Accumulate sequence lines
                    seq += line.upper()
            # Save the last accession and sequence
            if accession and seq:
                sequences[accession] = seq
        
        logger.debug(f"Read {len(sequences)} sequence(s) from {fasta_path.name}")
    except FileNotFoundError:
        logger.error(f"File not found: {fasta_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading FASTA file {fasta_path}: {e}")
        raise
    
    return sequences

def reverse_complement(seq: str) -> str:
    """
    Returns the reverse complement of a DNA sequence.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Reverse complement of the input sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    rev_comp = ''.join(complement.get(base, 'N') for base in reversed(seq))
    return rev_comp

# Standard genetic code codon table
codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def translate_sequence(seq: str) -> str:
    """
    Translates a nucleotide sequence into a protein sequence using the codon_table.
    
    Args:
        seq: Nucleotide sequence string
        
    Returns:
        Translated protein sequence (stop codons removed)
    """
    protein = ''
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa = codon_table.get(codon, 'X')  # 'X' if codon is invalid
        protein += aa
    return protein.rstrip('*')  # Remove trailing stop codon '*'

def find_orfs_in_seq(seq: str, min_length: int = 300) -> List[Tuple[int, int, str, str]]:
    """
    Finds all ORFs in the forward strand of a sequence.
    
    Args:
        seq: DNA sequence string
        min_length: Minimum ORF length in nucleotides
        
    Returns:
        List of tuples containing (start, end, strand, ORF sequence)
    """
    orfs = []
    seq_len = len(seq)
    for frame in range(3):
        i = frame
        while i < seq_len - 2:
            codon = seq[i:i+3]
            if codon == 'ATG':
                # Found a start codon
                for j in range(i+3, seq_len-2, 3):
                    stop_codon = seq[j:j+3]
                    if stop_codon in ['TAA', 'TAG', 'TGA']:
                        # Found a stop codon
                        orf_seq = seq[i:j+3]
                        if len(orf_seq) >= min_length:
                            orfs.append((i, j+3, '+', orf_seq))
                        i = j + 3
                        break
                else:
                    i += 3
            else:
                i += 3
    return orfs

def find_orfs_in_seq_reverse(seq: str, min_length: int = 300) -> List[Tuple[int, int, str, str]]:
    """
    Finds all ORFs in the reverse strand of a sequence.
    
    Args:
        seq: DNA sequence string
        min_length: Minimum ORF length in nucleotides
        
    Returns:
        List of tuples containing (start, end, strand, ORF sequence)
    """
    rev_seq = reverse_complement(seq)
    orfs = []
    seq_len = len(rev_seq)
    for frame in range(3):
        i = frame
        while i < seq_len - 2:
            codon = rev_seq[i:i+3]
            if codon == 'ATG':
                # Found a start codon
                for j in range(i+3, seq_len-2, 3):
                    stop_codon = rev_seq[j:j+3]
                    if stop_codon in ['TAA', 'TAG', 'TGA']:
                        orf_seq = rev_seq[i:j+3]
                        if len(orf_seq) >= min_length:
                            # Convert rev_seq indexes back to the original forward-seq coordinates if needed
                            start = len(seq) - j - 3
                            end = len(seq) - i
                            orfs.append((start, end, '-', orf_seq))
                        i = j + 3
                        break
                else:
                    i += 3
            else:
                i += 3
    return orfs

def find_orfs_all(seq: str, min_length: int = 300) -> List[Tuple[int, int, str, str]]:
    """
    Finds all ORFs in both forward and reverse strands.
    
    Args:
        seq: DNA sequence string
        min_length: Minimum ORF length in nucleotides
        
    Returns:
        List of tuples containing (start, end, strand, ORF sequence)
    """
    orfs = find_orfs_in_seq(seq, min_length)
    orfs += find_orfs_in_seq_reverse(seq, min_length)
    return orfs

def _write_protein_fasta(out_handle, header: str, protein_seq: str) -> None:
    """
    Writes the protein sequence to an output file, wrapping at 60 characters.
    
    Args:
        out_handle: File handle to write to
        header: FASTA header (should include '>')
        protein_seq: Protein sequence string
    """
    out_handle.write(header + '\n')
    for i in range(0, len(protein_seq), 60):
        out_handle.write(protein_seq[i:i+60] + '\n')


def _process_single_fasta(fasta_path: Path, min_length: int = 300) -> Dict[str, str]:
    """
    Process a single FASTA file and return protein sequences.
    
    Args:
        fasta_path: Path object to the FASTA file
        min_length: Minimum ORF length in nucleotides
        
    Returns:
        Dictionary with protein IDs as keys (e.g., '>accession_1') and protein sequences as values
    """
    results = {}
    sequences = read_fasta(fasta_path)
    
    for accession, seq in sequences.items():
        orfs = find_orfs_all(seq, min_length=min_length)
        logger.debug(f"Found {len(orfs)} ORF(s) in '{accession}'")
        
        for orf_count, orf in enumerate(orfs, start=1):
            start, end, strand, orf_seq = orf
            protein_seq = translate_sequence(orf_seq)
            protein_id = f">{accession}_{orf_count}"
            results[protein_id] = protein_seq
    
    logger.info(f"Extracted {len(results)} protein(s) from {fasta_path.name}")
    return results


def process_fasta(fasta_path: Path, min_length: int = 300) -> Dict[str, str]:
    """
    Process a single FASTA file and extract protein sequences from ORFs.
    
    Args:
        fasta_path: Path object to the FASTA file (DNA sequences)
        min_length: Minimum ORF length in nucleotides (default: 300)
        
    Returns:
        Dictionary with protein IDs as keys and sequences as values
        
    Example:
        >>> from pathlib import Path
        >>> fasta_file = Path('data/raw/phages/sample.fasta')
        >>> proteins = process_fasta(fasta_file)
        >>> # Returns: {'>seq1_1': 'MPKMP...', '>seq1_2': 'MPRN...'}
    """
    # Convert to Path object if string is passed
    fasta_path = Path(fasta_path)
    
    if not fasta_path.exists():
        logger.error(f"File does not exist: {fasta_path}")
        raise FileNotFoundError(f"File does not exist: {fasta_path}")
    
    if not fasta_path.is_file():
        logger.error(f"Path is not a file: {fasta_path}")
        raise ValueError(f"Path is not a file: {fasta_path}")
    
    if fasta_path.suffix.lower() not in ['.fasta', '.fa', '.fna', '.fsa']:
        logger.warning(f"File {fasta_path.name} does not have a typical FASTA extension")
    
    return _process_single_fasta(fasta_path, min_length)


def save_protein_sequences(protein_data: Dict[str, str], output_dir: Path, output_filename: str) -> Path:
    """
    Save protein sequences to a FASTA file.
    
    Args:
        protein_data: Dictionary with protein IDs as keys and sequences as values
                     (output from process_fasta)
        output_dir: Path object for output directory (required)
        output_filename: Name for the output file (e.g., 'sample_protein.fasta')
        
    Returns:
        Path object to the saved file
        
    Example:
        >>> from pathlib import Path
        >>> fasta_file = Path('data/raw/phages/sample.fasta')
        >>> proteins = process_fasta(fasta_file)
        >>> output_dir = Path('data/processed/datasets')
        >>> output_file = save_protein_sequences(proteins, output_dir, 'sample_protein.fasta')
        >>> # Saves to: data/processed/datasets/sample_protein.fasta
    """
    # Convert to Path object if needed
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create full output path
    output_path = output_dir / output_filename
    
    with open(output_path, 'w') as f:
        for protein_id, protein_seq in protein_data.items():
            _write_protein_fasta(f, protein_id, protein_seq)
    
    logger.info(f"Saved {len(protein_data)} protein(s) to {output_filename}")
    return output_path

