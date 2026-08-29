# k-mer Based Genetic Diagnostics of _MUC1_: A Privacy-First Approach

This repository contains the computational code and workflows developed for the MSc thesis:

> **k-mer Based Genetic Diagnostics of _MUC1_: A Privacy-First Approach**

## Project overview

This project investigates **k-mer-based approaches for genetic diagnostics of _MUC1_**, with a particular focus on the generalisation of machine-learning models across different data domains.

The project evaluates several settings:

- **Within-domain evaluation**
- **Domain transfer**
- **Domain adaptation**
- **Domain-adversarial learning**

The project also investigates the use of **privacy-preserving representations**, including hashed k-mer representations, and evaluates their potential within the proposed diagnostic framework.

To support these analyses, the repository contains the computational workflows for:

- preparing genomic reference data;
- generating positive and negative simulated samples;
- simulating sequencing reads;
- constructing haplotype-mixture (diploid-like) samples;
- processing _MUC1_ reads;
- generating k-mer feature representations; and
- performing the downstream machine-learning analyses.

A conventional **VNtyper2** analysis is also included as a baseline for comparison with the k-mer-based machine-learning approach.

## Computational workflow

The overall computational workflow using the simulated data is:

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
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
K-mer feature generation  VNtyper2
        │                 baseline
        ▼
Machine-learning
analyses
        │
        └─────────┬─────────┘
                  ▼
              Comparison
```

For simulated data, the processed _MUC1_ reads generated during the data-generation workflow provide the input for the downstream analyses.

For machine-learning analyses involving real sequencing data, the same feature-generation workflow can be applied directly to the relevant real sequencing data. Depending on the experimental setting, machine-learning analyses may therefore use simulated data, real data, or combinations of both.

For simulated data, the k-mer representation is generated from the processed reads and used by the machine-learning analyses. VNtyper2 is applied independently as a conventional baseline where required.

For details of the data-generation workflow, including the alternative routes for processing simulated reads, see [`data_generation/README.md`](data_generation/README.md).

## Repository structure

```text
.
├── analysis/
│   ├── baseline/
│   ├── methodological_analyses/
│   ├── pipelines/
│   └── targeted_analyses/
│
├── data_generation/
│   ├── reference_preparation/
│   ├── sample_preparation/
│   ├── read_simulation/
│   └── sample_generation/
│
├── feature_generation/
│
├── environment/
│
├── CITATION.cff
├── LICENSE
└── README.md
```

### `data_generation/`

Contains the workflow used to prepare the reference, generate positive and negative samples, simulate sequencing reads, construct haplotype-mixture (diploid-like) samples, and produce processed _MUC1_ reads for the downstream analyses.

See [`data_generation/README.md`](data_generation/README.md) for details of the data-generation workflow.

### `feature_generation/`

Contains the scripts used to convert the processed sequencing reads into **k-mer feature tables** used by the machine-learning analyses.

### `analysis/`

Contains the downstream analyses, including:

- main machine-learning pipelines;
- targeted analyses;
- methodological analyses; and
- the VNtyper2 baseline.

See [`analysis/README.md`](analysis/README.md) for details of the individual analyses.

### `environment/`

Contains Conda environment specifications and documentation of the software and computational requirements used by the project.

Not all software used in the workflow is contained within the Conda environment files. Some external tools are installed separately and invoked directly by the workflow scripts.

See [`environment/README.md`](environment/README.md) for environment setup, external software requirements, and computational dependencies.

## Reproducibility

The computational workflows were developed and executed in a **Linux-based, SLURM-managed computing environment**.

Bash scripts are used to execute computational steps and, where appropriate, **SLURM job arrays** are used to process multiple samples in parallel. Some scripts act as wrappers around Python programs or external tools, while others implement computational workflows directly.

The repository contains the workflow scripts, environment specifications, and documentation supporting reproduction of the computational analyses. Some external software and reference datasets are not distributed with the repository; their requirements and versions are documented in [`environment/README.md`](environment/README.md).

Generated sequencing data, intermediate files, and analysis outputs are not stored in the repository.

## Getting started

The general workflow for reproducing the computational analysis is:

1. **Set up the required software and computational environments.**  
   See [`environment/README.md`](environment/README.md).

2. **Prepare the reference data.**  
   See `data_generation/reference_preparation/`.

3. **Prepare positive and negative sequence material.**  
   See `data_generation/sample_preparation/`.

4. **Simulate sequencing reads.**  
   See `data_generation/read_simulation/`.

5. **Generate haplotype-mixture (diploid-like) samples where required.**  
   See `data_generation/sample_generation/`.

6. **Filter _MUC1_ reads.**  
   The `filter_muc1_reads.sh` workflow can be applied either directly to the simulated reads or to reads generated from the haplotype-mixture (diploid-like) samples. The resulting processed reads provide the input to the downstream analyses.

7. **Generate k-mer feature tables.**  
   See `feature_generation/`.

8. **Run the machine-learning analyses and VNtyper2 baseline.**  
   See [`analysis/README.md`](analysis/README.md).

The individual workflow scripts contain the commands and SLURM configuration required for their respective computational steps.

## Project context

This repository provides the computational implementation supporting the MSc thesis.

The dissertation provides the detailed methodological rationale, experimental design, parameter choices, results, and interpretation. This repository is intended to provide the corresponding computational workflows and reproducibility information.

## Data

The repository contains the code required to generate the simulated data and to perform the downstream analyses using simulated and, where applicable, real sequencing data.

Generated sequencing data, intermediate files, and analysis outputs are not included in the repository. Reference data and other external inputs that are required for particular workflows are documented separately where applicable.

## Citation

If using the code or workflows from this repository, please refer to [`CITATION.cff`](CITATION.cff).

## License

The code in this repository is released under the [MIT License](LICENSE).
