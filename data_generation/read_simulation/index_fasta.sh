#!/bin/bash

# =============================================
# Script: index_fasta
# Purpose: Index fasta sequences
# Author: NS
# =============================================

#SBATCH --job-name=faidx
#SBATCH --output=/faidx_%A_%a.out
#SBATCH --error=/faidx_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --array=1-70

set -euo pipefail

DEBUG_LOG="/faidx_debug3.log"
mkdir -p "$(dirname "$DEBUG_LOG")"

start_all=$(date +%s)

module load samtools

OUTPUT_DIR="../WTdata2"
FILES=($OUTPUT_DIR/*.fa)

# Check if there are files to process
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No FASTA files found in $OUTPUT_DIR" >> "$DEBUG_LOG"
    exit 1
fi

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
    idx=$((SLURM_ARRAY_TASK_ID-1))
    echo "Running command: samtools faidx \"${FILES[$idx]}\""
    echo "=================================================="
} >> "$DEBUG_LOG"

# Exit if no file for this task ID
if [[ -z "${FILES[$idx]}" ]]; then
    echo "No file for task ID $SLURM_ARRAY_TASK_ID, exiting." >> "$DEBUG_LOG"
    exit 0
fi

samtools faidx "${FILES[$idx]}"

end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Total runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
