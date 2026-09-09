#!/bin/bash

# =============================================
# Script: submit_snakemake
# Purpose: Run Snakemake with SLURM cluster support
# Author: NS
# =============================================

#SBATCH --job-name=snakemake_pipeline
#SBATCH --output=slurmLog/snakemake_%j.log
#SBATCH --error=slurmLog/snakemake_%j.err
#SBATCH --time=24:15:00  # Make sure the --time is long enough for the entire pipeline to finish
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
   

# ============================================================
# Environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
cd /home/username/NEAT-3.4
conda activate neat34

export NEAT_PYTHON="$CONDA_PREFIX/bin/python"

module load snakemake/9.19.0/gcc-py312


# ============================================================
# Diagnostics
# ============================================================

echo "=================================================="
echo "Snakemake production run"
echo "=================================================="

echo "Hostname:          $(hostname)"
echo "Date:              $(date)"
echo "Snakemake:         $(snakemake --version)"
echo "NEAT_PYTHON:       $NEAT_PYTHON"
echo "Python:            $($NEAT_PYTHON --version)"
echo "Python executable: $($NEAT_PYTHON -c 'import sys; print(sys.executable)')"

echo "=================================================="

# ============================================================
# Run Snakemake
# ============================================================

start_all=$(date +%s)

# Run Snakemake with SLURM cluster support
snakemake \
    --snakefile ../workflow/Snakefile \
    --profile ../workflow/slurm_profile \
    all

status=$?

# ============================================================
# Runtime
# ============================================================
end_all=$(date +%s)
total=$((end_all - start_all))
minutes=$(( total / 60 ))
seconds=$(( total % 60 ))

echo "=================================================="
echo "✅ Total runtime: ${minutes} min ${seconds} sec"
echo "=================================================="
