#!/bin/bash

#SBATCH --job-name=kmc_job
#SBATCH --output=/kmerSimUnmergedDip_c200_1.log
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:10:00


module load kmc/3.2.4-mcmesu/gcc

start_all=$(date +%s)


kmc_tmp_dir="kmc_tmp"
mkdir -p "$kmc_tmp_dir"

# Define datasets and k-mer lengths
declare -A datasets
datasets["MUC1_simneg"]="../diploidNegSharked"
datasets["MUC1_simpos"]="../diploidPosSharked12"

kmer_lengths=(23 31)

# Loop over datasets
for dataset_name in "${!datasets[@]}"; do
    input_dir="${datasets[$dataset_name]}"

    # Assign type column value
    if [[ "$dataset_name" == "MUC1_simneg" ]]; then
        dataset_type="neg"
    elif [[ "$dataset_name" == "MUC1_simpos" ]]; then
        dataset_type="pos"
    else
        dataset_type="unknown"
    fi

    # Loop over k-mer lengths
    for k in "${kmer_lengths[@]}"; do
        output_csv="kmer${dataset_name^}Unmerged${k}.csv" # Capitalize dataset name
        echo "ID,kmer_seq,count,type" > "$output_csv" # Write CSV header

        # Loop over each matching file (read1 or read2)
	for input_file in "$input_dir"/*_read[12].fq; do
            filename=$(basename "$input_file") # Extract full filename without path
            id=$(echo "$filename" | sed -E 's/_read[12]\.fq$//')
            db_name="kmc_db_${id}_k${k}" # Temporary DB name based on ID

            # Run KMC
            kmc -k"$k" -ci5 -t6 -fq "$input_file" "$db_name" "$kmc_tmp_dir"

            # Dump and process KMC output
            kmc_tools transform "$db_name" dump tmp_kmer.txt
            # Append processed kmers with ID to the CSV
            awk -v id="$id" -v type="$dataset_type" '{ print id "," $1 "," $2 "," type }' tmp_kmer.txt >> "$output_csv"

            # Cleanup temporary db
            rm -f "${db_name}"* tmp_kmer.txt
        done

        echo "Finished: $dataset_name, k=$k -> $output_csv"

        # Deduplicate each dataset-specific CSV
        echo "Deduplicating $output_csv..."
        # Create a temporary file to store the deduplicated data
        tmp_file=$(mktemp)

        # Use awk to process the CSV:
        #   -F',' sets the input field separator to comma (for CSV)
        #   OFS=',' sets the output field separator to comma
        awk -F',' '
        # If it is the first line (header), print it and skip to the next line
        NR == 1 { print; next }
        {
            # Create a unique key for each row using ID, kmer_seq, and type
            # FS is the field separator (comma)
            key = $1 FS $2 FS $4  # ID + kmer_seq + type
            # Sum the count for this key
            count[key] += $3
        }
        # After processing all lines, print the deduplicated data
        END {
            for (k in count) {
#                print k, count[k] 	# prints integer instead of char ("pos"/"neg")
		        # Split the key back into its parts: ID, kmer_seq, type
                # n is the number of parts (not used here, but required by split)
                n = split(k, parts, FS)
                # Print: ID, kmer_seq, summed_count, type
		        print parts[1], parts[2], count[k], parts[3]
            }
        }' OFS=',' "$output_csv" > "$tmp_file" && mv "$tmp_file" "$output_csv"
    done
done

# Cleanup temp directory and all its contents
# This directory was used by KMC to store intermediate files
trap 'rm -rf "$kmc_tmp_dir"' EXIT   # cleaned up directory, even if the script fails midway 

# === Final step: Combine CSVs by k-mer length ===
# Loop over each k-mer length (e.g., 23, 31)
for k in "${kmer_lengths[@]}"; do
    # Define the name of the combined output file for this k-mer length
    combined_file="kmerCombinedUnmerged${k}.csv"
    # Write the CSV header to the combined file
    echo "ID,kmer_seq,count,type" > "$combined_file"

    # Loop over each dataset (e.g., MUC1_simneg, MUC1_simpos)
    for dataset_name in "${!datasets[@]}"; do
        # Construct the filename for this dataset and k-mer length
        file="kmer${dataset_name^}Unmerged${k}.csv"
        # If the file exists, append all lines except the header to the combined file
        if [[ -f "$file" ]]; then
            tail -n +2 "$file" >> "$combined_file"
        fi
    done

    echo "Created combined file: $combined_file"

done

echo "All k-mers processed and saved"

end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Total runtime: ${minutes} min ${seconds} sec"
echo "=================================================="

