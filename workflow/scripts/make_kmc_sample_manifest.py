#!/usr/bin/env python3

import argparse
import csv


parser = argparse.ArgumentParser()

parser.add_argument("--positive-manifest", required=True)
parser.add_argument("--negative-manifest", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()

with open(args.output, "w", newline="\n") as out:
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(["ID", "R1", "R2", "type"])

    for manifest, sample_type in [
        (args.positive_manifest, "pos"),
        (args.negative_manifest, "neg")]:

        with open(manifest, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                writer.writerow([
                    row["ID"].strip(),
                    row["R1"].strip(),
                    row["R2"].strip(),
                    sample_type])
