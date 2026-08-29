#!/bin/bash

# =============================================
# Script: gff3_to_bed
# Purpose: Extract exons annotations from gff into bed
# Author: NS
# =============================================

#SBATCH --job-name=gff3_to_bed
#SBATCH --output=/g2b_%A_%a.out
#SBATCH --error=/g2b_%A_%a.err
#SBATCH --time=00:01:00                  
#SBATCH --mem=2G                         
#SBATCH --array=1-70

# Path to your Python script
SCRIPT="../gff_to_bed.py"

# Directory containing your GFF files
GFF_DIR="../WTdata2"

# Directory for BED output
BED_DIR="../WTdata2"

# Create output directory if it doesn't exist
mkdir -p "$BED_DIR"

# List of GFF files (one per line)
FILE_LIST=($GFF_DIR/*.gff.gz)

# Get the file for this array task
GFF_FILE=${FILE_LIST[$SLURM_ARRAY_TASK_ID-1]}

# Check if file exists
if [ ! -f "$GFF_FILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Set output BED file name
BED_FILE=$BED_DIR/$(basename $GFF_FILE .gff.gz).bed

start_all=$(date +%s)

source ~/miniconda3/etc/profile.d/conda.sh
cd /home/username/NEAT-3.4
conda activate neat34

# Run the script
python $SCRIPT $GFF_FILE $BED_FILE

conda deactivate


end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
