"""
Requires Python 3.8, Biopython, pandas, and gzip.
"""

import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import gzip
import os
import pandas as pd
from typing import List


def extractWholeChromosome(seq_file: str, chr_name: str) -> SeqRecord:
    """
    Extract the entire sequence of the specified chromosome or contig from a FASTA file.
    """
    with gzip.open(seq_file, "rt") as fseq:
        for record in SeqIO.parse(fseq, "fasta"):
            if record.id == chr_name or record.id.endswith(chr_name):
                return record
    print(f"Chromosome or Contig {chr_name} not found in {seq_file}.")
    return None

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <fasta_path> <contig_id> <output_path>")
        sys.exit(1)
    fasta_path, contig_id, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    contig_record = extractWholeChromosome(fasta_path, contig_id)
    if contig_record:
        with open(output_path, "w") as f_out:
        #with gzip.open(output_path, "wt") as f_out:    
            SeqIO.write(contig_record, f_out, "fasta")
        print(f"Contig {contig_id} saved to {output_path}")
    else:
        sys.exit(1)
