"""
Requires Python 3.8, Biopython, pandas, and gzip.
"""

import gzip
import sys
import os
import pandas as pd
from typing import Dict, Set


def extract_annotations(gff_file: str, contig_id: str, output_file: str) -> None:
    """
    Extract annotations for a single contig from a GFF file.
    Preserves headers and comments, skips duplicates.
    Output is saved as a gzipped file.
    """
    wanted = {contig_id}
    seen_headers = set()

    with gzip.open(gff_file, 'rt') as fin, gzip.open(output_file, 'wt') as fout:
        for line in fin:
            if line.startswith('#'):
                if line not in seen_headers:
                    fout.write(line)
                    seen_headers.add(line)
                continue
            if not line.strip() or line.strip() == '###':
                continue
            fields = line.split('\t')
            if fields and fields[0] in wanted:
                fout.write(line)

def main(gff_file: str, contig_id: str, output_file: str) -> None:
    """
    Extract annotations for a single contig from a GFF file.
    """
    print(f"Extracting contig {contig_id} from {gff_file} to {output_file}")
    extract_annotations(gff_file, contig_id, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract GFF annotations for a single contig.")
    parser.add_argument("gff_file", help="Path to the GFF file")
    parser.add_argument("contig_id", help="Contig ID to extract")
    parser.add_argument("output_file", help="Path to the output GFF file")
    args = parser.parse_args()

    main(args.gff_file, args.contig_id, args.output_file)

