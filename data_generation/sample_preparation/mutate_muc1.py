"""
Requires Python 3.8, Biopython, pandas, and gzip.
"""

import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import re
import gzip
import random
from typing import List, Tuple, Optional, Dict
import os


def print_usage_guide():
    """Print a detailed usage guide for the script."""
    guide = """
=== MUC1 Motif Mutation Script: Usage Guide ===

This script searches for motifs in the MUC1 gene region, applies a random mutation,
and outputs the results in various formats.

---
### Basic Usage
python script.py <seq_file> <gff_file> [options]

- <seq_file>: Path to the input FASTA file (can be gzipped).
- <gff_file>: Path to the input GFF file (can be gzipped).

---
### Main Options
--motif-file <file>       Path to the motif TSV file (default: VNTR_typology.tsv).
--motif-found-files <file> Path to the motif found log file (default: motif_found_files_16.txt).
--summary-table <file>    Path to the summary table CSV. If not provided, no summary table will be written.
--dry-run                 Only search for motifs, do not apply mutations.
--output <file>           Path to the output FASTA file for mutated sequences (default: output_mutated.fasta.gz).
--bed-output <file>       Path to the output BED file. If not provided, no BED file will be written.
--gene-region-output <file>
                          Path to the output FASTA file for the mutated gene region. If not provided, no gene region file will be written.

---
### Example Commands
1. Dry run (only search for motifs):
   python script.py seq.fa gff.gff --dry-run

2. Apply mutation and output only the mutated FASTA:
   python script.py seq.fa gff.gff --output mutated.fa

3. Apply mutation and output all files:
   python script.py seq.fa gff.gff --output mutated.fa --bed-output exons.bed --gene-region-output gene.fa

---
### Notes
- If --bed-output or --gene-region-output are not provided, those files will not be written.
- The script will print progress and output file paths to stdout.
"""
    print(guide)

def load_motifs(motif_file: str) -> Dict[str, str]:
    """Load motifs from a TSV file."""
    motifs = {}
    with open(motif_file, 'rt') as vntr:
        for line in vntr:
            motif_name, motif_sequence = line.strip().split()
            motifs[motif_name] = motif_sequence
    return motifs

def extract_all_records(seq_file: str) -> List[SeqRecord]:
    """Extract all records from a FASTA file (possibly gzipped)."""
    open_func = gzip.open if seq_file.endswith(".gz") else open
    with open_func(seq_file, "rt") as fseq:
        return list(SeqIO.parse(fseq, "fasta"))

def get_muc1_coords(gff_file: str, feature_name: str = "MUC1") -> Tuple[str, int, int, str]:
    """Parse the GFF file and return the chromosome, start, end coordinates, and strand of the MUC1 gene."""
    open_func = gzip.open if gff_file.endswith(".gz") else open
    with open_func(gff_file, "rt") as f:
        for line in f:
            if f"Name={feature_name}" in line:
                fields = line.strip().split("\t")
                chr_name, start, end, strand = fields[0], int(fields[3]) - 1, int(fields[4]), fields[6]
                return chr_name, start, end, strand
    raise ValueError(f"Feature {feature_name} not found in {gff_file}")

def extract_region(record: SeqRecord, start: int, end: int, strand: str, padding: int = 5000) -> SeqRecord:
    """Extract a region from a SeqRecord, with optional padding."""
    start = max(0, start - padding)
    end = min(len(record.seq), end + padding)
    subseq = record.seq[start:end]
    if strand == "-":
        subseq = subseq.reverse_complement()
    return SeqRecord(subseq, id=f"{record.id}:{start}-{end}", description=record.description)
    #return SeqRecord(subseq, id=f"{record.id}:{start}:{end}", description=record.description)

def find_motif(record: SeqRecord, motif: str) -> List[Tuple[int, int]]:
    """Find all occurrences of a motif in a SeqRecord."""
    return [(m.start(), m.end()) for m in re.finditer(motif, str(record.seq))]

def apply_mutation(record: SeqRecord, s: int, e: int, mutseq: str) -> SeqRecord:
    """Apply a mutation to a SeqRecord at the given coordinates."""
    seq_list = list(str(record.seq))
    seq_list[s:e] = list(mutseq)
    mutated_seq = ''.join(seq_list)
    return SeqRecord(Seq(mutated_seq), id=record.id, description=record.description)

def write_summary_table(
    sample_info: List[str],
    motif_counts: Dict[str, int],
    summary_table: str
) -> None:
    """Write summary table with motif counts."""
    header = ["Sample", "chr", "start", "end", "length"] + ["n_" + k for k in motif_counts.keys()]
    file_exists = os.path.isfile(summary_table)
    with open(summary_table, "a") as table_out:
        if not file_exists or os.path.getsize(summary_table) == 0:
            table_out.write("\t".join(header) + "\n")
        row = sample_info + [str(motif_counts.get(m, 0)) for m in motif_counts.keys()]
        table_out.write("\t".join(row) + "\n")

def write_motif_log(chr_name: str, muc1_start: int, muc1_end: int, strand: str,
                    motif_results: Dict[str, List[Tuple[int, int]]], sample_name: str, motif_found_files: str) -> None:
    """Write motif search results to log file (appends if file exists)."""
    with open(motif_found_files, "a") as out:
        out.write(f"=== Sample: {sample_name} | MUC1 Region: {chr_name}:{muc1_start}-{muc1_end} (strand: {strand}) ===\n")
        #out.write(f"=== Sample: {sample_name} | MUC1 Region: {chr_name}:{muc1_start}:{muc1_end} (strand: {strand}) ===\n")
        for motif_name, positions in motif_results.items():
            if positions:
                out.write(f"Motif {motif_name} found at positions: {positions}\n")
            else:
                out.write(f"Motif {motif_name} not found.\n")

def gff3_exons_to_bed_dicts(gff3_path: str) -> List[Dict]:
    """
    Parse a GFF3 file, extract exon features, and return a list of BED entries as dictionaries.

    Args:
        gff3_path: Path to the input GFF3 file.

    Returns:
        List of dictionaries, each representing an exon with keys: chrom, start, end, exon_id, strand.
    """
    bed_entries = []
    open_func = gzip.open if gff3_path.endswith(".gz") else open
    with open_func(gff3_path, "rt") as gff:
   # with open(gff3_path, 'r') as gff:
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

#def write_full_bed(bed_dicts: list[dict], bed_path: str) -> None:
def write_full_bed(bed_dicts: List[Dict], bed_path: str) -> None:
    """Write all exons to a BED file."""
    with open(bed_path, "w") as f:
        for exon in bed_dicts:
            f.write(f"{exon['chrom']}\t{exon['start']}\t{exon['end']}\t{exon['exon_id']}\t0\t{exon['strand']}\n")

def amend_bed_for_mutation(
    bed_dicts: List[Dict],
    chr_name: str,
    gene_start: int,
    gene_end: int,
    bed_path: str
) -> None:
    """
    For each exon on the correct chromosome and within the gene region,
    extend its start and end by 10.
    Write all exons to the output BED file.
    """
    amended_exons = []
    for exon in bed_dicts:
        if (
            exon["chrom"] == chr_name
            and not (exon["end"] < gene_start or exon["start"] > gene_end)
        ):
            # Extend start and end by 10
            exon["start"] = max(0, exon["start"] - 10)  # Ensure not negative
            exon["end"] += 10
        amended_exons.append(exon)

    # Save amended BED
    with open(bed_path, "w") as f:
        for exon in amended_exons:
            f.write(f"{exon['chrom']}\t{exon['start']}\t{exon['end']}\t{exon['exon_id']}\t0\t{exon['strand']}\n")

def save_mutated_gene_region(
    mutated_record: SeqRecord,
    gene_start: int,
    gene_end: int,
    description: str,
    output_path: str
) -> None:
    """Save the mutated gene region to a FASTA file."""
    subseq = mutated_record.seq[gene_start:gene_end]
    subrecord = SeqRecord(
        subseq,
        id=f"{mutated_record.id}",
        description=description
    )
    open_func = gzip.open if output_path.endswith(".gz") else open
    with open_func(output_path, "wt") as f:
        SeqIO.write([subrecord], f, "fasta")

def main(args):
    try:
        # Define dictionaries for mutations and motifs
#        d_mutations = {
#            "X": {"Xatt": "GCCCACGGTGTCACCTCGGCCCCGGACACCAGGCCGGCCCCGGGCTCCACCGCCCCCCCA"},
#            "5": {"5att": "GCCCACGATGTCACCTCAGCCCCGGACAACAAGCCAGCCCCGGGCTCCACCGCCCCCCCA"},
#            "B": {"Batt": "GCCCACGGTGTCACCTCGGCCCCGGAGAGCAGGCCGGCCCCGGGCTCCACCGCCCCCCCCA"}
#        }
#        d_mutations = {
#            "27pos": {
#                "27dupC": "GGGCTCCACCGCCCCCCCCAGCCCACGGTGTC",
#                "27insCCCC": "GGGCTCCACCGCCCCCCCCCCCAGCCCACGGTGTC",
#                "26_27insG": "GGGCTCCACCGCCCCCCGCAGCCCACGGTGTC",
#                "28dupA": "GGGCTCCACCGCCCCCCCAAGCCCACGGTGT",
#                "23delinsAT": "GGCTCCACCGCCATCCCCAGCCCACGGTGTC"
#            }
#        }
#        d_mutations = {
#            "27pos": {
#                "27dupC": "GGGCTCCACCGCCCCCCCCAGCCCACGGTGTC"
#            }
#        }
        d_mutations = {
            "X": {"Xatt": "GCCCACGGTGTCACCTCGGCCCCGGACACCAGGCCGGCCCCGGGCTCCACCGCCCCCCCA"},
            "5": {"5att": "GCCCACGATGTCACCTCAGCCCCGGACAACAAGCCAGCCCCGGGCTCCACCGCCCCCCCA"},
            "B": {"Batt": "GCCCACGGTGTCACCTCGGCCCCGGAGAGCAGGCCGGCCCCGGGCTCCACCGCCCCCCCCA"},
            "27pos": {
                "27dupC": "GGGCTCCACCGCCCCCCCCAGCCCACGGTGTC",
                "27insCCCC": "GGGCTCCACCGCCCCCCCCCCCAGCCCACGGTGTC",
                "26_27insG": "GGGCTCCACCGCCCCCCGCAGCCCACGGTGTC",
                "28dupA": "GGGCTCCACCGCCCCCCCAAGCCCACGGTGT",
                "23delinsAT": "GGCTCCACCGCCATCCCCAGCCCACGGTGTC"
            }
        }

        # Load motifs and mutations
        d_motifs = load_motifs(args.motif_file)
        d_motifs.update({"27pos": "GGGCTCCACCGCCCCCCCAGCCCACGGTGTC"})

        # Extract all records
        records = extract_all_records(args.seq_file)
        if not records:
            raise ValueError("No records found in the FASTA file.")

        # Get MUC1 coordinates
        try:
            chr_name, muc1_start, muc1_end, strand = get_muc1_coords(args.gff_file)
        except ValueError as e:
            raise ValueError(f"Error in GFF file: {e}")

        # Find the record matching the chromosome/contig
        record = next((r for r in records if r.id == chr_name), None)
        if not record:
            raise ValueError(f"Chromosome/contig {chr_name} not found in the FASTA file.")

        # Search for motifs in the MUC1 region
        muc1_region = extract_region(record, muc1_start, muc1_end, strand, padding=5000)
        motif_results = {mn: find_motif(muc1_region, seq) for mn, seq in d_motifs.items()}
        motifs_found = any(motif_results.values())

        # Log and summarize
        # write_motif_log(chr_name, muc1_start, muc1_end, strand, motif_results, args.seq_file, args.motif_found_files)
        if args.motif_found_files:
            write_motif_log(
                chr_name, muc1_start, muc1_end, strand,
                motif_results, args.seq_file, args.motif_found_files
            )
        if args.summary_table:
            write_summary_table(
                [args.seq_file, chr_name, str(muc1_start), str(muc1_end), str(muc1_end - muc1_start)],
                {mn: len(pos) for mn, pos in motif_results.items()},
                args.summary_table
            )

        if not motifs_found:
            print("No motifs found in the MUC1 region.")
        elif args.dry_run:
            print("Dry run complete. Motifs found (see motif_found_files.txt). No mutations applied.")
        else:
            # Only consider motifs that are in both d_motifs and d_mutations
            valid_motifs = [mn for mn in d_motifs.keys() if mn in d_mutations and motif_results[mn]]
            if valid_motifs:
                # Randomly select one motif and one position
                selected_motif = random.choice(valid_motifs)
                positions = motif_results[selected_motif]
                s, e = random.choice(positions)
                s_whole = s + (muc1_start - 5000)
                e_whole = e + (muc1_start - 5000)
                mutname, mutseq = random.choice(list(d_mutations[selected_motif].items()))
                
                # Apply mutation
                mutated_record = apply_mutation(record, s_whole, e_whole, mutseq)
                mutated_record.id = record.id
                mutated_record.description = f"{record.description} | Mutated: {selected_motif} -> {mutname} at {s_whole}-{e_whole}"

                # Replace the mutated record in the list
                mutated_records = [mutated_record if r.id == record.id else r for r in records]

                # Write all records to output
                root, ext = os.path.splitext(args.output)
                if ext == '.gz':
                    inner_root, inner_ext = os.path.splitext(root)
                    ext = inner_ext + ext
                    root = inner_root
                mut_output = f"{root}_{selected_motif}_{mutname}_at_{s_whole}-{e_whole}{ext}"
                open_func = gzip.open if mut_output.endswith(".gz") else open
                with open_func(mut_output, "wt") as f:
                    SeqIO.write(mutated_records, f, "fasta")
                print(f"Mutated sequences saved to {mut_output}")

                # Only write BED file if --bed-output is provided
                if args.bed_output:
                    bed_dicts = gff3_exons_to_bed_dicts(args.gff_file)
                    amend_bed_for_mutation(
                        bed_dicts,
                        chr_name,
                        muc1_start,
                        muc1_end,
                        args.bed_output
                    )
                    print(f"Amended BED file saved to {args.bed_output}")

                # Only write gene region file if --gene-region-output is provided
                if args.gene_region_output:
                    gene_root, gene_ext = os.path.splitext(args.gene_region_output)
                    if gene_ext == '.gz':
                        inner_gene_root, inner_gene_ext = os.path.splitext(gene_root)
                        gene_ext = inner_gene_ext + gene_ext
                        gene_root = inner_gene_root
                    mutated_gene_description = f"{mutated_record.description} | Gene_region: {muc1_start}-{muc1_end}"
                    gene_region_output = f"{gene_root}_{selected_motif}_{mutname}_gene_region_{s_whole}-{e_whole}{gene_ext}"
                    save_mutated_gene_region(mutated_record, muc1_start, muc1_end, mutated_gene_description, gene_region_output)
                    print(f"Mutated gene region saved to {gene_region_output}")

    except Exception as e:
        print(f"ERROR: {e}")
        print("For help, run: python script.py --usage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for motifs in MUC1 region, randomly apply one mutation, and save all contigs with the mutation applied."
    )
    parser.add_argument("seq_file", help="Path to the FASTA file (can be gzipped).")
    parser.add_argument("gff_file", help="Path to the GFF file (can be gzipped).")
    parser.add_argument("--motif-file", default="/scratch/solankin/data/genomes/simulation/MUC1_VNTR_typology.tsv", help="Path to the motif TSV file.")
    parser.add_argument("--motif-found-files", default=None, help="Path to the motif found log file. If not provided, no motif found log will be written.")
    parser.add_argument("--summary-table", default=None, help="Path to the summary table CSV. If not provided, no summary table will be written.")
    parser.add_argument("--dry-run", action="store_true", help="Only search for motifs, do not apply mutations.")
    parser.add_argument("--output", default="output_mutated.fasta.gz", help="Path to the output FASTA file for mutated sequences.")
    parser.add_argument("--bed-output", default=None, help="Path to the output BED file. If not provided, no BED file will be written.")
    parser.add_argument("--gene-region-output", default=None, help="Path to the output FASTA file for the mutated gene region. If not provided, no gene region file will be written.")
    parser.add_argument("--usage", action="store_true", help="Print this usage guide and exit.")
    args = parser.parse_args()

    if args.usage:
        print_usage_guide()
    else:
        main(args)

