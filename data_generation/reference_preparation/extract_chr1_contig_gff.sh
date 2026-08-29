#!/bin/bash

# =============================================
# Script: extract_chr1_contig_gff
# Purpose: Extract chr1/chr1 MUC1 contig annotation
# Author: NS
# =============================================

#SBATCH --job-name=extract_chr1
#SBATCH --output=/extract_chr1_gff_%A_%a.out
#SBATCH --error=/extract_chr1_gff_%A_%a.err
#SBATCH --time=00:03:00           
#SBATCH --mem=12G                 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                   
#SBATCH --array=1-395


# Get the line corresponding to the array task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../matched_paths_with_contig.txt)
GFF=$(echo $LINE | awk '{print $2}')
CONTIG=$(echo $LINE | awk '{print $3}')


# Extract prefix from GFF filename
PREFIX=$(basename "$GFF" .gff3.gz)

# Set output directory and prefix
OUTPUT_DIR="../chr1data2"
mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/${PREFIX}_chr1.gff.gz"

JOB_NAME="$PREFIX"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home/username/NEAT-3.4
conda activate neat34


start_all=$(date +%s)

python ../extract_chr1_contig_gff.py "$GFF" "$CONTIG" "$OUTPUT_FILE"

conda deactivate


end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} (${PREFIX}) runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
