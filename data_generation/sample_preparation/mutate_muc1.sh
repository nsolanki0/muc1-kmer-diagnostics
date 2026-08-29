#!/bin/bash

# =============================================
# Script: mutate_muc1
# Purpose: Mutate MUC1 sequences
# Author: NS
# =============================================

#SBATCH --job-name=mutate_muc1
#SBATCH --output=/mutate_muc1_%A_%a.out
#SBATCH --error=/mutate_muc1_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-40
 
set -euo pipefail

DEBUG_LOG="../muc1_var_debug2.log"
mkdir -p "$(dirname "$DEBUG_LOG")"

start_all=$(date +%s)

source ~/miniconda3/etc/profile.d/conda.sh
cd /home/username/NEAT-3.4
conda activate neat34

# Paths
LIST_FILE="../4muc1M_chr1FaGff_pairs5.txt"
MOTIF_FILE="../MUC1_VNTR_typology.tsv"
OUTPUT_DIR="../MUC1_variants2"
LOG_FILE="../all_MUC1_detect2.csv"

mkdir -p "$OUTPUT_DIR"

# Extract the nth line from the list file
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
SEQ_FILE=$(echo "$LINE" | awk '{print $1}')
GFF_FILE=$(echo "$LINE" | awk '{print $2}')

if [[ ! -f "$SEQ_FILE" || ! -f "$GFF_FILE" ]]; then
    echo "Error: Input file not found for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

# Extract basename for output files
BASE_FA=$(basename "$SEQ_FILE" .fa)
BASE_GFF=$(basename "$GFF_FILE" .gff.gz)

# Output paths (uncompressed)
OUTPUT_FA="$OUTPUT_DIR/${BASE_FA}.fa"
BED_OUTPUT="$OUTPUT_DIR/${BASE_GFF}_muc1M_exons.bed"
GENE_OUTPUT="$OUTPUT_DIR/${BASE_FA}_muc1_gene_region.fa.gz"


# Print to debug log
{
    echo "=================================================="
    echo "Slurm job ID: $SLURM_JOB_ID"
    echo "Job name: $SLURM_JOB_NAME"
    echo "Array task ID: $SLURM_ARRAY_TASK_ID"
    REQUESTED_TIME=$(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -oP 'TimeLimit=\K[^ ]+') 
    echo "Requested time: $REQUESTED_TIME"
    echo "Requested memory: ${SLURM_MEM_PER_NODE:-not set}"
    echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-not set}"
    echo "Running command: ../mutate_muc1.py \"$SEQ_FILE\" \"$GFF_FILE\" --motif-file \"$MOTIF_FILE\" --summary-table \"$LOG_FILE\" --output \"$OUTPUT_FA\" --bed-output \"$BED_OUTPUT\" --gene-region-output \"$GENE_OUTPUT\""
    echo "=================================================="
} >> "$DEBUG_LOG"


# Run the script
python ../mutate_muc1.py "$SEQ_FILE" "$GFF_FILE" \
  --motif-file "$MOTIF_FILE" \
  --summary-table "$LOG_FILE" \
  --output "$OUTPUT_FA" \
  --bed-output "$BED_OUTPUT" \
  --gene-region-output "$GENE_OUTPUT"

conda deactivate

end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Total runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
