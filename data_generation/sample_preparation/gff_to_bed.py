"""
Requires Python 3.8, Biopython, pandas, and gzip.
"""

from typing import List, Dict
import gzip


def gff3_exons_to_bed_dicts(gff3_path: str) -> List[Dict]:
    bed_entries = []
    open_func = gzip.open if gff3_path.endswith(".gz") else open
    with open_func(gff3_path, "rt") as gff:
        for line in gff:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            feature_type = fields[2]
            if feature_type.lower() != 'exon':
                continue
            chrom, start, end, strand, attrs = fields[0], fields[3], fields[4], fields[6], fields[8]
            exon_id = ""
            for attr in attrs.split(';'):
                if attr.startswith('exon_id='):
                    exon_id = attr.split('exon_id=')[1]
                    break
            if exon_id:
                bed_entries.append({
                    "chrom": chrom,
                    "start": int(start)-1,
                    "end": int(end),
                    "exon_id": exon_id,
                    "strand": strand
                })
    return bed_entries

def write_full_bed(bed_dicts: List[Dict], bed_path: str) -> None:
    with open(bed_path, "w") as f:
        for exon in bed_dicts:
            f.write(f"{exon['chrom']}\t{exon['start']}\t{exon['end']}\t{exon['exon_id']}\t0\t{exon['strand']}\n")

if __name__ == "__main__":
    import sys
    gff3_path = sys.argv[1]
    bed_path = sys.argv[2]
    exons = gff3_exons_to_bed_dicts(gff3_path)
    write_full_bed(exons, bed_path)
