#!/usr/bin/env python3

import argparse
import os
import sys


parser = argparse.ArgumentParser(
    description="Validate sharked paired-end FASTQ files"
)

parser.add_argument("--input-manifest", required=True)
parser.add_argument("--output-manifest", required=True)

args = parser.parse_args()


os.makedirs(
    os.path.dirname(os.path.abspath(args.output_manifest)),
    exist_ok=True
)

valid_pairs = 0
invalid_pairs = 0

with open(args.input_manifest) as infile, \
     open(args.output_manifest, "w") as outfile:

    header = infile.readline().strip()

    outfile.write("ID\tR1\tR2\n")

    for line in infile:
        line = line.strip()

        if not line:
            continue

        fields = line.split("\t")

        if len(fields) < 3:
            print(
                f"WARNING: malformed manifest line: {line}",
                file=sys.stderr
            )
            continue

        sample_id, r1, r2 = fields[:3]

        r1_exists = os.path.isfile(r1)
        r2_exists = os.path.isfile(r2)

        r1_nonempty = r1_exists and os.path.getsize(r1) > 0
        r2_nonempty = r2_exists and os.path.getsize(r2) > 0

        if r1_nonempty and r2_nonempty:
            outfile.write(
                f"{sample_id}\t{r1}\t{r2}\n"
            )
            valid_pairs += 1

        else:
            invalid_pairs += 1

            print(
                f"INVALID PAIR: {sample_id}",
                file=sys.stderr
            )

            print(
                f"  R1 exists: {r1_exists}, "
                f"size: {os.path.getsize(r1) if r1_exists else 0}",
                file=sys.stderr
            )

            print(
                f"  R2 exists: {r2_exists}, "
                f"size: {os.path.getsize(r2) if r2_exists else 0}",
                file=sys.stderr
            )

            # Remove both files if either member of the pair is invalid.
            if r1_exists:
                os.remove(r1)

            if r2_exists:
                os.remove(r2)


print(
    f"Validation complete: "
    f"{valid_pairs} valid pairs, "
    f"{invalid_pairs} invalid pairs",
    file=sys.stderr
)
