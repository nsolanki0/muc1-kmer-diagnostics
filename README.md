# k-mer Based Genetic Diagnostics of MUC1: A Privacy-First Approach

This repository contains the computational code and workflows developed
for the MSc thesis:

**k-mer Based Genetic Diagnostics of MUC1: A Privacy-First Approach**

The project investigates k-mer-based approaches for genetic diagnostics
of MUC1, with a particular focus on model generalisation across domains,
including within-domain evaluation, domain transfer, domain adaptation,
and domain-adversarial approaches.

The repository also contains the workflows developed for generating
simulated datasets, preparing reference genomic data, and automating
the computational analyses using Bash and SLURM.

The thesis also investigates the potential of privacy-preserving representations 
within this framework, including the use of hashed representations. The repository 
contains the corresponding implementation and analysis.


MUC1 k-mer-based analysis
This repository contains the computational workflow for generating simulated MUC1 sequencing data and evaluating k-mer-based machine-learning approaches for analysis of the resulting samples.
The workflow consists of two main stages:

Data generation — preparation of reference sequences, generation of positive and negative samples, read simulation, and processing of simulated reads.
Analysis — generation of k-mer features, application of machine-learning approaches, and comparison against VNtyper2 as a conventional analysis baseline.

Workflow
The overall workflow is:
Reference genome
      ↓
Reference preparation
      ↓
Positive / negative sequence preparation
      ↓
Read simulation
      ↓
Diploid sample generation
      ↓
Filter MUC1 reads
      ↓
   ┌──┴──────────────┐
   ↓                 ↓
k-mer generation    VNtyper2
   ↓                 ↓
ML analyses       baseline
   └───────┬─────────┘
           ↓
       Comparison

The processed reads produced during data generation provide the input for both downstream approaches. The k-mer representation is generated from these processed reads for use by the machine-learning analyses, while VNtyper2 is applied independently as a conventional baseline.
Repository structure
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
├── environment/
├── CITATION.cff
└── LICENSE

data_generation/
Contains the workflow used to prepare the reference, generate positive and negative samples, simulate sequencing reads, construct haplotype mixtures (diploid like) samples, and produce the processed reads used by the downstream analyses.
See data_generation/README.md for details.

feature_generation/
Contains the scripts used to convert the processed sequencing data into k-mer feature tables for the machine-learning analyses.
analysis/
Contains the downstream analyses, including the main machine-learning pipelines, targeted and methodological analyses, and the VNtyper2 baseline.
See analysis/README.md for details.

environment/
Contains the environment specifications required by the different computational tools and analyses. The environments are documented separately to keep software and dependency information independent from the workflow descriptions.

Reproducibility
The workflow is designed to run on a SLURM-based computing environment. Several Bash scripts act as execution wrappers around Python programs or external tools and use SLURM arrays to process multiple samples.
The SLURM implementation is part of the individual computational workflows rather than being treated as a separate automation layer.

Software environments used by the workflow are specified in environment/.

## Getting started

1. Clone the repository.
2. Set up the required computational environments using the instructions in `environment/`.
3. Prepare the reference data using `data_generation/reference_preparation/`.
4. Generate the simulated samples using `data_generation/`.
5. Generate k-mer features using `feature_generation/`.
6. Run the analyses in `analysis/`.

Data
The repository contains the code required to generate and analyse the simulated data. Generated sequencing data, intermediate files, and analysis outputs are not included in the repository unless explicitly stated.

Citation
If using this repository or the associated methods, please refer to CITATION.cff.