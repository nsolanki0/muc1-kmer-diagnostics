#!/usr/bin/env python3

import argparse
import os


def validate(input_manifest, output_manifest):

    valid = 0
    invalid = 0

    with open(input_manifest) as infile, \
         open(output_manifest, "w") as outfile:

        infile.readline()

        outfile.write("Pair_ID\tR1\tR2\n")

        for line in infile:

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 5:
                continue

            pair_id = fields[0]

            # Actual diploid output names
            output_dir = args.output_dir

            r1 = os.path.join(
                output_dir,
                f"{pair_id}_read1.fq.gz"
            )

            r2 = os.path.join(
                output_dir,
                f"{pair_id}_read2.fq.gz"
            )

            if not os.path.isfile(r1):
                print(f"Missing R1: {r1}")
                invalid += 1
                continue

            if not os.path.isfile(r2):
                print(f"Missing R2: {r2}")
                invalid += 1
                continue

            if os.path.getsize(r1) == 0:
                print(f"Empty R1: {r1}")
                invalid += 1
                continue

            if os.path.getsize(r2) == 0:
                print(f"Empty R2: {r2}")
                invalid += 1
                continue

            outfile.write(
                f"{pair_id}\t{r1}\t{r2}\n"
            )

            valid += 1

    print(f"Valid pairs: {valid}")
    print(f"Invalid pairs: {invalid}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    validate(
        args.input,
        args.output
    )
