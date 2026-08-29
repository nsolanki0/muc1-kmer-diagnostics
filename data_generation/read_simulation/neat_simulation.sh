#!/bin/bash

# =============================================
# Script: neat_simulation
# Purpose: Simulate reads using NEAT-3.4
# Author: NS
# =============================================

#SBATCH --job-name=neat34_sim
#SBATCH --output=/np27dupC_c200_%A_%a.out
#SBATCH --error=/np27dupC_c200_%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --array=1-27


# Get the line corresponding to the array task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../4neat_Mv27dupC_bedFa_files.txt)
BED=$(echo $LINE | awk '{print $1}')
FASTA=$(echo $LINE | awk '{print $2}')

# Extract prefix from FASTA filename
FASTA_NAME=$(basename "$FASTA" .fa)
PREFIX=$(echo "$FASTA_NAME" | sed 's/Homo_sapiens-//; s/-softmasked//')

# Set output directory and prefix
OUTPUT_DIR="../neat_pos27dupC_c200"
mkdir -p "${OUTPUT_DIR}"

JOB_NAME="$PREFIX"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home/username/NEAT-3.4
conda activate neat34

start_all=$(date +%s)


python gen_reads.py -R 150 --pe 400 40 -c 200 \
-r "$FASTA" \
-tr "$BED" \
-o "${OUTPUT_DIR}/$PREFIX" \
--bam --vcf -M 0

conda deactivate

end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} (${PREFIX}) runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
