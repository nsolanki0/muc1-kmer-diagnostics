Computational Environment
This directory contains the Conda environment specifications and documentation for the software and computational requirements used in the project.
The computational workflows were developed and executed in a Linux-based, SLURM-managed high-performance computing (HPC) environment. Several workflows use dedicated Conda or mamba environments to accommodate software with different Python or R version requirements.

Software environments
The project uses the following programming languages and software environments:
Python 3.8.20 — used for compatibility with NEAT (3.4) and for custom sequence-processing scripts involving FASTQ and GFF/GFF3 files.
Python 3.11.13 — used for the main machine-learning workflows.
Python 3.11.15 — used for the domain-adversarial neural network (DANN) implementation.
Python 3.13.7 — used for the VNtyper2 (2.0, Kestrel) baseline analysis.
R 4.3.2 — used for statistical analysis and data visualisation.
mamba — used to manage the Sourmash environment.
The corresponding environment specifications are provided in this directory where applicable.
Bioinformatics software
The computational workflows use a combination of environment-managed software and standalone bioinformatics tools.
The principal tools and versions used in the project include:

Tool	Version	Purpose
NEAT	3.4	Sequencing-read simulation
DWGSIM	0.1.17-dev	Sequencing-read simulation
seqtk	1.5	Sequence processing
Shark	1.2.0	MUC1 read extraction
KMC	3.2.4	k-mer counting
BWA	0.7.18	Sequence alignment
SAMtools	1.23.1	BAM/SAM processing
MUMmer	4.0.0beta2	Genome alignment
minimap2	2.28	Sequence alignment
Sourmash	4.9.4	Sequence similarity and sketching
VNtyper	2.0 (Kestrel)	MUC1 baseline analysis

Some of these tools are managed through dedicated Conda/mamba environments, while others are installed separately and invoked directly by the workflow scripts.
Workflow automation and HPC
Bash scripting is used for workflow automation, file management, and SLURM job submission.
Computationally intensive workflows are executed on a SLURM-managed HPC system, with SLURM job arrays used where appropriate to process independent samples or computational tasks in parallel.

Some Bash scripts act as wrappers around Python programs or external bioinformatics tools, while others contain the computational workflow and SLURM configuration directly.

External software and licensing
The MIT License in the root of this repository applies to the original code and workflow scripts distributed in this repository.
Third-party software used by the workflows is not relicensed under the MIT License. Each external tool remains subject to its own license and associated terms.

Users are responsible for consulting the respective software documentation and license information when installing or using third-party tools. The versions used for the analyses are documented above to support reproducibility.

External software is not necessarily distributed with this repository.

Reproducibility
Before running the workflows, ensure that the required software versions, reference data, and computational resources are available.
The environment specifications in this directory provide the basis for recreating the computational environments used in the project. However, some standalone external tools must be installed separately.

The exact commands and software usage are documented in the relevant workflow scripts and directory-level README files.

The computational environment described here corresponds to the software configuration used for the analyses reported in the accompanying MSc thesis.