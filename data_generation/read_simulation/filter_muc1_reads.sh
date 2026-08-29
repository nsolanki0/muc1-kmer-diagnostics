#!/bin/bash

# =============================================
# Script: shark_reads
# Purpose: Shark reads to filter MUC1 sequence
# Author: NS
# =============================================

#SBATCH --job-name=shark_reads            
#SBATCH --output=/shark_n_c200_hap12_%A_%a.out
#SBATCH --error=/shark_n_c200_hap12_%A_%a.err
#SBATCH --time=00:03:00                  
#SBATCH --mem=32G                        
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4  
#SBATCH --array=1-166

# Get the line corresponding to the array task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../4shark_Nc200_12_r1r2files.txt)

R1=$(echo $LINE | awk '{print $1}')
R2=$(echo $LINE | awk '{print $2}')

# Extract prefix from R1 filename (remove _read1.fq.gz)
JOB_NAME=$(basename "$R1" _read1.fq.gz)

# Define suffixes
R1_SUFFIX="_sharked_read1.fq"
R2_SUFFIX="_sharked_read2.fq"
SSV_SUFFIX=".ssv"

# Set output directory and prefix
OUTPUT_DIR="../shark_hap_neg_c200_12"

mkdir -p "${OUTPUT_DIR}"

# Construct output filenames
R1_NAME="${JOB_NAME}${R1_SUFFIX}"
R2_NAME="${JOB_NAME}${R2_SUFFIX}"
SSV_NAME="${JOB_NAME}${SSV_SUFFIX}"

module load shark/1.2.0/gcc

start_all=$(date +%s)


# Run shark
shark -r ../muc1_seqs.fasta \
-1 "${R1}" \
-2 "${R2}" \
-o "${OUTPUT_DIR}/${R1_NAME}" \
-p "${OUTPUT_DIR}/${R2_NAME}" > "${OUTPUT_DIR}/${SSV_NAME}"


end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} (${JOB_NAME}) runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
