#!/usr/bin/env python3

import argparse
import os


def read_sample_list(filename):
    with open(filename) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")]

def find_negative_pair(negative_dir, sample):
    r1 = os.path.join(negative_dir, f"{sample}_chr1_read1.fq.gz")
    r2 = os.path.join(negative_dir, f"{sample}_chr1_read2.fq.gz")

    if not os.path.isfile(r1):
        raise RuntimeError(f"Missing R1: {r1}")

    if not os.path.isfile(r2):
        raise RuntimeError(f"Missing R2: {r2}")

    return r1, r2

def add_pair(outfile, seen, r1a, r2a, r1b, r2b):
    base_a = os.path.basename(r1a).replace("_read1.fq.gz", "")
    base_b = os.path.basename(r1b).replace("_read1.fq.gz", "")

    pair_id = f"{base_a}_x_{base_b}"

    if pair_id in seen:
        raise RuntimeError(f"Duplicate Pair_ID: {pair_id}")

    seen.add(pair_id)

    outfile.write(
        f"{pair_id}\t"
        f"{r1a}\t{r1b}\t"
        f"{r2a}\t{r2b}\n")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--negative-mat", required=True)
    parser.add_argument("--negative-pat", required=True)

    parser.add_argument("--negative-mat2", required=True)
    parser.add_argument("--negative-pat2", required=True)

    parser.add_argument("--negative-dir", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    neg_mat = read_sample_list(args.negative_mat)
    neg_pat = read_sample_list(args.negative_pat)

    neg_mat2 = read_sample_list(args.negative_mat2)
    neg_pat2 = read_sample_list(args.negative_pat2)

    seen = set()

    with open(args.output, "w") as outfile:
        outfile.write("Pair_ID\tR1_A\tR1_B\tR2_A\tR2_B\n")

        # NEG_MAT × NEG_PAT2
        for sample_a in neg_mat:
            r1a, r2a = find_negative_pair(args.negative_dir, sample_a)

            for sample_b in neg_pat2:
                r1b, r2b = find_negative_pair(args.negative_dir, sample_b)

                add_pair(outfile, seen, r1a, r2a, r1b, r2b)

        # NEG_PAT × NEG_MAT2
        for sample_a in neg_pat:
            r1a, r2a = find_negative_pair(args.negative_dir, sample_a)

            for sample_b in neg_mat2:
                r1b, r2b = find_negative_pair(args.negative_dir, sample_b)

                add_pair(outfile, seen, r1a, r2a, r1b, r2b)

    print(f"Created: {args.output}")
    print(f"Diploid pairs: {len(seen)}")


if __name__ == "__main__":
    main()
