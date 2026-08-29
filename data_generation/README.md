# Data Generation

This directory contains the computational workflows used to generate the simulated _MUC1_ sequencing data used in the downstream analyses.

The data-generation workflow prepares the genomic reference, generates positive and negative sequence material, simulates sequencing reads, constructs haplotype-mixture (diploid-like) samples, and processes the resulting reads for downstream analysis.

## Workflow

The data-generation workflow consists of several stages:

```text
Reference genome
      │
      ▼
Reference preparation
      │
      ▼
Positive / negative sequence preparation
      │
      ▼
Read simulation
      │
      ├───────────────────────┐
      │                       │
      ▼                       ▼
Diploid-like            Filter MUC1 reads
sample generation              │
      │                        │
      ▼                        │
Filter MUC1 reads            │
      │                        │
      └───────────┬────────────┘
                  ▼
           Processed reads
```

The processed reads produced at the end of this workflow are used by both downstream analysis approaches:

```text
                    Processed reads
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     K-mer feature generation      VNtyper2
              │                    baseline
              ▼
       Machine-learning
           analyses
```

## Directory structure

```text
data_generation/
├── reference_preparation/
├── sample_preparation/
├── read_simulation/
├── sample_generation/
└── README.md
```

### `reference_preparation/`

Contains scripts used to prepare the genomic reference data required for the simulation workflow.

The scripts include extraction and processing of the relevant chromosome 1 / _MUC1_ reference sequence and associated genomic annotation information.

### `sample_preparation/`

Contains scripts used to prepare the positive and negative sequence material used to generate the simulated samples.

This includes:

- generation of _MUC1_ sequence variants;
- conversion of genomic annotation information to BED format; and
- preparation of sequences required by the downstream simulation workflow.

The main mutation-generation workflow is implemented using `mutate_muc1.py` and its corresponding Bash/SLURM wrapper.

### `read_simulation/`

Contains scripts used to simulate sequencing reads and to filter/extract _MUC1_-related reads for downstream analysis.

The workflow includes:

- indexing the reference FASTA;
- sequencing-read simulation using NEAT; and
- filtering/extracting _MUC1_-related reads.

The `filter_muc1_reads.sh` workflow can be applied directly to real or simulated reads. For simulated data, it can also be applied to reads generated from the haplotype-mixture (diploid-like) sample-generation workflow, depending on the downstream analysis.

The resulting processed reads are used as input to both k-mer feature generation and the VNtyper2 baseline analysis.

### `sample_generation/`

Contains scripts used to construct the final haplotype-mixture (diploid-like) samples from simulated sequencing reads.

The workflows use SLURM job arrays to process multiple samples in parallel.

The `submit_positive.sh` and `submit_negative.sh` scripts are used for SLURM job submission and array management, while the corresponding sample-generation scripts perform the computational sample-generation workflow.

## Computational environment

The data-generation workflows were developed and executed in a Linux-based, SLURM-managed computing environment.

Several workflows use SLURM job arrays to process multiple samples in parallel. Bash scripts may act as wrappers around Python programs or external tools and may contain SLURM configuration required for batch execution.

Software environments and external tool requirements are documented in [`../environment/README.md`](../environment/README.md).

## Outputs

The main output of the data-generation workflow is a set of processed simulated sequencing reads representing the generated _MUC1_ samples.

These processed reads are subsequently used for:

1. **k-mer feature generation**, using the workflow in [`../feature_generation/`](../feature_generation/); and
2. **VNtyper2 baseline analysis**, using the workflow in [`../analysis/baseline/`](../analysis/baseline/).

Generated sequencing data and intermediate files are not stored in the repository.

## Reproducibility

The individual scripts contain the commands, parameters, and SLURM configuration required for their respective computational steps.

The main data-generation workflow proceeds through:

1. `reference_preparation/`
2. `sample_preparation/`
3. `read_simulation/`

Depending on the downstream analysis, simulated reads can either be filtered directly using `filter_muc1_reads.sh`, or first processed through `sample_generation/` to construct haplotype-mixture (diploid-like) samples before _MUC1_ read filtering.

Before running the workflows, ensure that the required software, reference data, and computational environments have been prepared. See [`../environment/README.md`](../environment/README.md).
