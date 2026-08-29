#!/bin/bash

# Check if directory is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/gff_files"
    exit 1
fi

dir="$1"
output_file="MUC1_contigs_260223.txt"

# Header
echo -e "GFF_file_path\tContig" > "$output_file"

# Loop through each .gff.gz file in the specified directory
for file in "$dir"/*.gff3.gz; do
    # Use zcat and grep to find MUC1 (whole word), extract contig, and append to output
    zcat "$file" | grep -w "MUC1" | cut -f1 | while read -r contig; do
        echo -e "$file\t$contig" >> "$output_file"
    done
done

echo "Done. Results saved to $output_file"
