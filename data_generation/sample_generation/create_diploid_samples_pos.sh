#!/bin/bash

# =============================================
# Script: create_diploid_samples_pos
# Purpose: Create diploid samples from haplotypes (.fq)
# Author: NS
# =============================================

#SBATCH --job-name=createDiploid
#SBATCH --output=/create_diploidP_%A_%a.out
#SBATCH --error=/create_diploidP_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-1681  # Adjust based on total pairs


## ADD OFFSET SO THAT EACH SUBBATCH STARTS WITH 1 AND ENDS WITH 1000, INSTEAD OF 1001-2000 (for eg.)
# Get the offset from the environment variable
OFFSET=${OFFSET:-0}
# Calculate the actual job index
ACTUAL_ID=$((OFFSET + SLURM_ARRAY_TASK_ID))
# Use ACTUAL_ID instead of SLURM_ARRAY_TASK_ID in the script
idx=$((ACTUAL_ID - 1))

module load seqtk/1.5/gcc

# For diploid positive samples, (i) Mat from positive and Pat from Negative,
# (ii) Pat from positive and Mat from Negative.

GROUP1_LIST="../44dipPosMatSharked_R1list.txt"
GROUP2_LIST="../44PdipNegPatSharked_R1list.txt"
OUTDIR="../diploidPosSharked2"

mkdir -p "$OUTDIR"

start_all=$(date +%s)


# Get the pair for this task by reading Group1 and Group2 sample names from files
mapfile -t GROUP1 < "$GROUP1_LIST"
mapfile -t GROUP2 < "$GROUP2_LIST"

# Calculate which pair to process
#idx=$((SLURM_ARRAY_TASK_ID - 1))
idx=$((ACTUAL_ID - 1))
rows=${#GROUP2[@]}
#rows=${#GROUP1[@]}
a_idx=$((idx / rows))
b_idx=$((idx % rows))

a="${GROUP1[$a_idx]}"
b="${GROUP2[$b_idx]}"

# Extract sample names (without read number)
a_sample=$(basename "$a" | sed 's/_read1.fq//')
b_sample=$(basename "$b" | sed 's/_read1.fq//')

# Construct read1 and read2 filenames
r1a="$a"
r2a="${a/_read1.fq/_read2.fq}"
r1b="$b"
r2b="${b/_read1.fq/_read2.fq}"

out_r1="$OUTDIR/${a_sample}_x_${b_sample}_read1.fq"
out_r2="$OUTDIR/${a_sample}_x_${b_sample}_read2.fq"

echo "Creating diploid sample: $a_sample x $b_sample"
echo "Using for read1: $r1a and $r1b"

# Subsample and combine R1
seqtk sample -s 42 "$r1a" 0.5 > "temp_${a_sample}_read1.fq"
seqtk sample -s 42 "$r1b" 0.5 > "temp_${b_sample}_read1.fq"
cat "temp_${a_sample}_read1.fq" "temp_${b_sample}_read1.fq" > "$out_r1"
rm "temp_${a_sample}_read1.fq" "temp_${b_sample}_read1.fq"

echo "Using for read2: $r2a and $r2b"

# Subsample and combine R2
seqtk sample -s 42 "$r2a" 0.5 > "temp_${a_sample}_read2.fq"
seqtk sample -s 42 "$r2b" 0.5 > "temp_${b_sample}_read2.fq"
cat "temp_${a_sample}_read2.fq" "temp_${b_sample}_read2.fq" > "$out_r2"
rm "temp_${a_sample}_read2.fq" "temp_${b_sample}_read2.fq"


end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Job ${SLURM_ARRAY_TASK_ID} (${PREFIX}) runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
