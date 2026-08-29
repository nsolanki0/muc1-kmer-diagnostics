#!/bin/bash

# Path to your original script
SCRIPT="../create_diploid_samples_neg.sh"

TOTAL_JOBS=6790

JOBS_PER_BATCH=1000

start=1
batch=1

while [ $start -le $TOTAL_JOBS ]; do
    end=$((start + JOBS_PER_BATCH - 1))
    if [ $end -gt $TOTAL_JOBS ]; then
        end=$TOTAL_JOBS
    fi
    jobs_in_batch=$((end - start + 1))
    echo "Submitting batch$batch: $jobs_in_batch jobs (array 1-$jobs_in_batch, offset $start)"
    sbatch --job-name=batch$batch --array=1-$jobs_in_batch --export=OFFSET=$((start - 1)) $SCRIPT
    start=$((end + 1))
    batch=$((batch + 1))
done

