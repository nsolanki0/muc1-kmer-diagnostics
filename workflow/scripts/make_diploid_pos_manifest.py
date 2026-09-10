#!/usr/bin/env python3

import argparse
import glob
import os


def read_sample_list(filename):
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")]

def find_positive_pair(positive_dir, sample):
    pattern = os.path.join(positive_dir, f"{sample}_chr1_*_read1.fq.gz")

    r1_files = sorted(glob.glob(pattern))

    if len(r1_files) != 1:
        raise RuntimeError(
            f"Expected exactly one positive R1 for {sample}; "
            f"found {len(r1_files)}")

    r1 = r1_files[0]
    r2 = r1.replace("_read1.fq.gz", "_read2.fq.gz")

    if not os.path.isfile(r2):
        raise RuntimeError(f"Missing R2: {r2}")

    return r1, r2

def find_negative_pair(negative_dir, sample):
    r1 = os.path.join(negative_dir, f"{sample}_chr1_read1.fq.gz")
    r2 = os.path.join(negative_dir, f"{sample}_chr1_read2.fq.gz")

    if not os.path.isfile(r1):
        raise RuntimeError(f"Missing R1: {r1}")

    if not os.path.isfile(r2):
        raise RuntimeError(f"Missing R2: {r2}")

    return r1, r2

def add_pair(outfile, seen, pos_r1, pos_r2, neg_r1, neg_r2):
    pos_base = os.path.basename(pos_r1).replace("_read1.fq.gz", "")
    neg_base = os.path.basename(neg_r1).replace("_read1.fq.gz", "")

    pair_id = f"{pos_base}_x_{neg_base}"

    if pair_id in seen:
        raise RuntimeError(f"Duplicate Pair_ID: {pair_id}")

    seen.add(pair_id)

    outfile.write(
        f"{pair_id}\t"
        f"{pos_r1}\t{neg_r1}\t"
        f"{pos_r2}\t{neg_r2}\n")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--positive-mat", required=True)
    parser.add_argument("--positive-pat", required=True)

    parser.add_argument("--negative-mat", required=True)
    parser.add_argument("--negative-pat", required=True)

    parser.add_argument("--positive-dir", required=True)
    parser.add_argument("--negative-dir", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    pos_mat = read_sample_list(args.positive_mat)
    pos_pat = read_sample_list(args.positive_pat)

    neg_mat = read_sample_list(args.negative_mat)
    neg_pat = read_sample_list(args.negative_pat)

    seen = set()

    with open(args.output, "w") as outfile:
        outfile.write("Pair_ID\tPOS_R1\tNEG_R1\tPOS_R2\tNEG_R2\n")

        # POS_MAT × NEG_PAT
        for pos_sample in pos_mat:
            pos_r1, pos_r2 = find_positive_pair(args.positive_dir, pos_sample)

            for neg_sample in neg_pat:
                neg_r1, neg_r2 = find_negative_pair(args.negative_dir, neg_sample)

                add_pair(outfile, seen, pos_r1, pos_r2, neg_r1, neg_r2)

        # POS_PAT × NEG_MAT
        for pos_sample in pos_pat:
            pos_r1, pos_r2 = find_positive_pair(args.positive_dir, pos_sample)

            for neg_sample in neg_mat:
                neg_r1, neg_r2 = find_negative_pair(args.negative_dir, neg_sample)

                add_pair(outfile, seen, pos_r1, pos_r2, neg_r1, neg_r2)

    print(f"Created: {args.output}")
    print(f"Diploid pairs: {len(seen)}")


if __name__ == "__main__":
    main()
