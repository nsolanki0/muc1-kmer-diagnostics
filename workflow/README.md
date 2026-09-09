# Snakemake Workflow

This directory contains the Snakemake workflow used to generate the simulated _MUC1_ sequencing data and the final k-mer feature tables.

The workflow is designed for execution on an HPC system using **Snakemake** with **SLURM**.

## Before running the workflow

Two prerequisites should be prepared before submitting the workflow.

**1. Copy the required Python scripts**
The Snakemake workflow uses Python scripts developed as part of the `data_generation` component of the project.

Before running the workflow, the required Python scripts should be available in the `scripts/` directory of the `workflow/` directory:

```text
workflow/scripts/
```

The scripts used by the workflow include:

```text
scripts/
├── extract_chr1_contig.py
├── extract_chr1_contig_gff.py
├── mutate_muc1.py
├── gff_to_bed.py
├── make_diploid_pos_manifest.py
├── make_diploid_neg_manifest.py
├── validate_diploid_pairs.py
├── validate_sharked_pairs.py
└── make_kmc_sample_manifest.py
```

The first four scripts originate from the `data_generation` component of the project; the remaining scripts support later stages of the Snakemake workflow.

The workflow expects these scripts to be available in `workflow/scripts/`.

**2. Set up the NEAT Python environment**
The workflow uses NEAT 3.4 for read simulation and requires the Python from the corresponding Conda environment.

The required environment must therefore be activated before running Snakemake, and `NEAT_PYTHON` must point to the Python executable in that environment.

The submission script performs this setup using:

```text
cd /home/username/NEAT-3.4
conda activate neat34

export NEAT_PYTHON="$CONDA_PREFIX/bin/python"
```

The detailed setup and explanation of the NEAT environment are provided in the [`../environment/README.md`](../environment/README.md).

## Running the workflow

The workflow uses relative paths based on the `workflow/` directory.

Therefore, the workflow should be run from the `workflow/` directory containing the required input data and workflow files.

The expected structure is:

```text
workflow/
├── Snakefile
├── submit_snakemake.sh
├── slurm_profile/
│   └── config.yaml
├── scripts/
├── resources/
├── gff3_release2/
├── data_release2/
└── ...
```

The exact contents of the input and output directories depend on the datasets being processed.

## Required input data

The pipeline requires matching FASTA and GFF files for each sample.

For example, a sample may consist of:

```text
GCA_041900145.1.unmasked.fa.gz
GCA_041900145.1.gff3.gz
```

The corresponding files should be placed in the appropriate input directories:

```text
workflow/
├── data_release2/
│   ├── GCA_041900145.1.unmasked.fa.gz
│   ├── GCA_041900165.1.unmasked.fa.gz
│   └── ...
│
└── gff3_release2/
    ├── GCA_041900145.1.gff3.gz
    ├── GCA_041900165.1.gff3.gz
    └── ...
```

The FASTA and GFF files are matched using their accession identifiers.

For example:

```text
GCA_041900145.1
```

identifies the pair:

```text
GCA_041900145.1.unmasked.fa.gz
GCA_041900145.1.gff3.gz
```

Both files are required for the corresponding sample to be processed.

## Sample lists

The `.txt` files in `resources/` contain sample identifiers rather than complete filenames.

For example:

```text
GCA_041900145.1
GCA_041900165.1
```

These identifiers correspond to the FASTA/GFF pairs described above.

The sample lists determine which samples are used for the different parts of the simulation workflow.

The current resource files include:

```text
resources/
├── positive_mat_list.txt
├── positive_pat_list.txt
├── negative_mat_list.txt
├── negative_pat_list.txt
├── negative_mat_list2.txt
├── negative_pat_list2.txt
├── MUC1_VNTR_typology.tsv
└── muc1_seqs.fasta
```

The two negative sample groups (`negative_*_list.txt` and `negative_*_list2.txt`) represent the two negative-sample categories used for data generation in the study.

The rationale for the construction of the positive and negative datasets, including the generation of diploid/child samples and the distinction between the two negative categories, is described in the accompanying MSc thesis.

## Workflow components

The main components are:

```text
workflow/
├── Snakefile
├── submit_snakemake.sh
├── slurm_profile/
│   └── config.yaml
├── scripts/
└── resources/
```

### `Snakefile`

Defines the complete Snakemake workflow, including:
- identification of MUC1-containing contigs
- matching FASTA and GFF files
- extraction of the relevant contig
- preparation of positive and negative samples
- MUC1 mutation generation
- NEAT read simulation
- generation of diploid read pairs
- validation of simulated read pairs
- Shark processing
- validation of Sharked reads
- KMC-based k-mer generation
- generation of k-mer feature tables

### `scripts/`

Contains the Python scripts used by individual workflow steps. Some of these scripts originate from the `data_generation` component of the project.

### `resources/`

Contains sample lists and reference/resource files required by the workflow.

### `slurm_profile/`

Contains the Snakemake SLURM profile:

```text
slurm_profile/
└── config.yaml
```

The profile specifies the SLURM executor and default cluster resources.

## SLURM execution

The workflow is launched using the submission script:

```text
submit_snakemake.sh
```

The script activates the required environment and loads Snakemake before starting the workflow.

The workflow is then run with:

```text
snakemake \
    --snakefile Snakefile \
    --profile slurm_profile \
    all
```

The SLURM profile is explicitly supplied using:

```text
--profile slurm_profile
```

The profile directory contains its own config.yaml, which controls Snakemake's interaction with SLURM.

> **Note:** The paths used within the Snakefile and submission script (e.g., `../workflow/resources/` or `../workflow/scripts/`) must be adjusted to the corresponding location of the `workflow/` directory on the HPC system.

The workflow can be submitted with:

```text
sbatch submit_snakemake.sh
```

## SLURM profile configuration

The current profile contains:

```text
executor: slurm

jobs: 100

default-resources:
    slurm_partition: std
    mem_mb: 8192
    runtime: 1440

latency-wait: 60
restart-times: 3
```

Individual workflow rules also specify their own resource requirements where appropriate.

## Outputs

The workflow generates intermediate files and manifests as it progresses through the pipeline.

The final targets are the k-mer feature tables for the two k-mer lengths (23 and 31):

```text
KmerTable/kmerCombinedUnmerged23.csv
KmerTable/kmerCombinedUnmerged31.csv
```

These contain the k-mer features, their counts, sample identifiers, and sample types (positive/negative) generated from the validated simulated samples.

## Methodological details

The workflow implements the simulation and preprocessing steps used in the study.

Detailed explanations of:

- the construction of positive and negative datasets
- the generation of diploid/child samples
- the rationale for the two negative-sample categories
- the biological and methodological motivation for the simulation design
- subsequent analytical use of the simulated datasets

are provided in the accompanying MSc thesis.
