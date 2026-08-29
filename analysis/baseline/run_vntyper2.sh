#!/bin/bash

# =============================================
# Script: run_vntyper2
# Purpose: Genotype MUC1 coding VNTR in ADTKD-MUC1
# Author: NS
# =============================================

#SBATCH --job-name=vntyper2            
#SBATCH --output=/vntyper_pc200_Conf_dip_unsharked_%A_%a.out
#SBATCH --error=/vntyper_pc200_Conf_dip_unsharked_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --mem=28G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-50

# Get the line corresponding to the array task ID
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../4vntyper_Pc200_diploidPosT2_unsharkedr1r2files.txt)

R1=$(echo $LINE | awk '{print $1}')
R2=$(echo $LINE | awk '{print $2}')

# Check if files exist
if [ ! -f "$R1" ] || [ ! -f "$R2" ]; then
    echo "Error: One or both input files do not exist."
    exit 1
fi

# Extract prefix from UNSHARKED R1 filename (remove _read1.fq.gz)
JOB_NAME=$(basename "$R1" _read1.fq.gz)

# Set output directory
DIR="../vntyper_Pc200_hapConf_diploidPosT2_unsharked"

OUTPUT_DIR="${DIR}/${JOB_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Log input and output
echo "Processing: $R1 and $R2"
echo "Output directory: $OUTPUT_DIR"

module purge

module load python/3.13.7 || { echo "Failed to load python module"; exit 1; }
module load fastp || { echo "Failed to load fastp module"; exit 1; }
module load samtools || { echo "Failed to load samtools module"; exit 1; }
module load bwa || { echo "Failed to load bwa module"; exit 1; }
module load java/11.0.24/openjdk || { echo "Failed to load java module"; exit 1; }


source $HOME/vntyper/bin/activate

start_all=$(date +%s)

# Print the command for debugging
echo "Running: vntyper --config-path ../vntyper/config.json pipeline --fastq1 $R1 --fastq2 $R2 --output-dir $OUTPUT_DIR --threads 4 --fast-mode"

vntyper --config-path ../vntyper/config.json pipeline \
    --fastq1 "${R1}" \
    --fastq2 "${R2}" \
    --output-dir "${OUTPUT_DIR}" \
    --threads 4 --fast-mode

end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} (${JOB_NAME}) runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
