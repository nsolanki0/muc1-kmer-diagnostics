#!/usr/bin/env python3

import argparse
import os
import sys


def validate(input_manifest, output_manifest, output_dir):

    valid = 0
    invalid = 0
    malformed = 0

    with open(input_manifest) as infile, \
         open(output_manifest, "w") as outfile:
    
        # Read and discard input header
        header = infile.readline()

        # Write validated output header
        outfile.write("Pair_ID\tR1\tR2\n")

        # Process every manifest entry
        for line_number, line in enumerate(infile, start=2):
            line = line.rstrip("\n")
            # Ignore completely blank lines
            if not line:
                continue
            fields = line.split("\t")

            # ------------------------------------------------
            # Manifest must contain:
            #
            # Pair_ID
            # POS_R1 / NEG_R1
            # NEG_R1 / NEG_R2
            # POS_R2 / ...
            # NEG_R2
            #
            # i.e. at least 5 columns.
            # ------------------------------------------------
            if len(fields) < 5:
                print(
                    f"MALFORMED MANIFEST LINE: "
                    f"line {line_number}: {line}",
                    file=sys.stderr)
                malformed += 1
                continue
            pair_id = fields[0].strip()

            # ------------------------------------------------
            # Empty Pair_ID is also malformed
            # ------------------------------------------------
            if not pair_id:
                print(
                    f"MALFORMED MANIFEST LINE: "
                    f"line {line_number}: empty Pair_ID",
                    file=sys.stderr)
                malformed += 1
                continue

            # Actual diploid output names
            r1 = os.path.join(output_dir, f"{pair_id}_read1.fq.gz")
            r2 = os.path.join(output_dir, f"{pair_id}_read2.fq.gz")

            # Check R1
            r1_exists = os.path.isfile(r1)
            r1_nonempty = (r1_exists and os.path.getsize(r1) > 0)

            # Check R2
            r2_exists = os.path.isfile(r2)
            r2_nonempty = (r2_exists and os.path.getsize(r2) > 0)

            # ------------------------------------------------
            # Pair is valid ONLY when both files exist
            # and both are non-empty.
            # ------------------------------------------------
            if r1_nonempty and r2_nonempty:
                outfile.write(f"{pair_id}\t{r1}\t{r2}\n")

                valid += 1

            # ------------------------------------------------
            # Pair is invalid.
            #
            # This is NOT a pipeline failure.
            # The pair is simply excluded.
            # ------------------------------------------------
            else:
                invalid += 1
                print(f"INVALID DIPLOID PAIR: {pair_id}", file=sys.stderr)

                print(
                    f"  R1 exists: {r1_exists}, "
                    f"size: "
                    f"{os.path.getsize(r1) if r1_exists else 0}",
                    file=sys.stderr)

                print(
                    f"  R2 exists: {r2_exists}, "
                    f"size: "
                    f"{os.path.getsize(r2) if r2_exists else 0}",
                    file=sys.stderr)

                # ------------------------------------------------
                # Remove BOTH files if either member is invalid.
                #
                # This prevents a partial pair from being reused
                # later in the pipeline.
                # ------------------------------------------------
                if r1_exists:
                    os.remove(r1)

                if r2_exists:
                    os.remove(r2)

    # --------------------------------------------------------
    # Final validation summary
    #
    # IMPORTANT:
    # malformed/invalid pairs do NOT cause a non-zero exit.
    # The script completes successfully as long as the
    # validation process itself succeeds.
    # --------------------------------------------------------
    print(
        f"Validation complete: "
        f"{valid} valid pairs, "
        f"{invalid} invalid pairs, "
        f"{malformed} malformed lines",
        file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate diploid paired-end FASTQ files")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    
    validate(args.input, args.output, args.output_dir)
