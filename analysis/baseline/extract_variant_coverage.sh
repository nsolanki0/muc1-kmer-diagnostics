#!/bin/bash

# =============================================
# Script: extract_variant_coverage
# Purpose: Extracts variant data and coverage info, by searching current directory and all its subdirectories
# Author: NS
# =============================================

# Output TSV file
output_file="../vntyper_c200_hapConf_diploidPosT2_unsharked_variant_cov_data.tsv"


# Write TSV header with custom coverage column names
echo -e "filename\tstep\tvariant\tmotif\tpos\tref\talt\tvariant_status\tC_mean\tC_median\tC_stdev\tC_min\tC_max\tC_region_length\tC_uncovered_bases\tC_percent_uncovered" > "$output_file"

# Loop through all matching files
find ../vntyper_Pc200_hapConf_diploidPosT2_unsharked -type f -name "pipeline_summary.json" | while read json_file; do
    # Get filename (basename only)
    filename=$(basename "$(dirname "$json_file")")

    # Extract Kestrel Genotyping data
    jq -c '.steps[]? | select(.step == "Kestrel Genotyping") | .parsed_result.data[]?' "$json_file" | while read entry; do
        variant=$(echo "$entry" | jq -r '.Variant')
        motif=$(echo "$entry" | jq -r '.Motif')
        pos=$(echo "$entry" | jq -r '.POS')
        ref=$(echo "$entry" | jq -r '.REF')
        alt=$(echo "$entry" | jq -r '.ALT')

        # Determine if it's a real or "None" variant
        if [[ "$variant" == "None" ]]; then
            variant_status="None"
        else
            variant_status="Real"
        fi

        # Extract Coverage Calculation data for the same file
        coverage_data=$(jq -c '.steps[]? | select(.step == "Coverage Calculation") | .parsed_result.data[0]' "$json_file")
        C_mean=$(echo "$coverage_data" | jq -r '.mean // ""')
        C_median=$(echo "$coverage_data" | jq -r '.median // ""')
        C_stdev=$(echo "$coverage_data" | jq -r '.stdev // ""')
        C_min=$(echo "$coverage_data" | jq -r '.min // ""')
        C_max=$(echo "$coverage_data" | jq -r '.max // ""')
        C_region_length=$(echo "$coverage_data" | jq -r '.region_length // ""')
        C_uncovered_bases=$(echo "$coverage_data" | jq -r '.uncovered_bases // ""')
        C_percent_uncovered=$(echo "$coverage_data" | jq -r '.percent_uncovered // ""')

        # Write row to TSV
        echo -e "$filename\tKestrel Genotyping\t$variant\t$motif\t$pos\t$ref\t$alt\t$variant_status\t$C_mean\t$C_median\t$C_stdev\t$C_min\t$C_max\t$C_region_length\t$C_uncovered_bases\t$C_percent_uncovered" >> "$output_file"
    done
done

echo "✅ Done. Output saved to $output_file"
