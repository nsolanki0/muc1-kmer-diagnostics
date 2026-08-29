# Computational Environment

This directory contains the Conda environment specifications and documentation for the software and computational requirements used in the project.

The computational workflows were developed and executed in a Linux-based, SLURM-managed high-performance computing (HPC) environment. Several workflows use dedicated Conda or mamba environments to accommodate different software and version requirements.

## Software Environments

The project uses the following programming languages and software environments:

| Software | Version | Purpose |
|---|---|---|
| Python | 3.8.20 | Used for compatibility with NEAT (3.4) and for custom sequence-processing scripts involving FASTQ and GFF/GFF3 files. |
| Python | 3.11.13 | Used for the main machine-learning workflows. |
| Python | 3.11.15 | Used for the domain-adversarial neural network (DANN) implementation. |
| Python | 3.13.7 | Used for the VNtyper2 (2.0, Kestrel) baseline analysis. |
| R | 4.3.2 | Used for statistical analysis and data visualisation. |
| mamba | — | Used to manage the Sourmash environment. |

Dedicated environment specifications are provided in this directory where applicable.

## Environment Setup

The environment specification files provided in this directory can be used to recreate the computational environments used for the project.

### Conda

If Conda is available, an environment can generally be created from an environment specification using:

```bash
conda env create -f <environment_file>.yml
```

The environment can then be activated using:

```bash
conda activate <environment_name>
```

### Mamba

For environments managed using mamba, an environment can be created using:

```bash
mamba env create -f <environment_file>.yml
```

The environment can then be activated using:

```bash
mamba activate <environment_name>
```

Replace <environment_file>.yml and <environment_name> with the corresponding environment file and environment name provided in this directory.

After creating an environment, verify the relevant software versions before running the corresponding workflow.

For example:

```bash
python --version
```

or, where applicable:

```bash
R --version
```

Some software used in the project is installed separately from the Conda environments. These tools must be installed and configured independently before running workflows that require them.

Bioinformatics Software
The computational workflows use a combination of environment-managed software and standalone bioinformatics tools.

The principal tools and versions used in the project are:

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

Some tools are managed through dedicated Conda or mamba environments, while others are installed separately and invoked directly by the workflow scripts.

External Software and Licensing
The MIT License in the root of this repository applies to the original code and workflow scripts distributed in this repository.
Third-party software used by the workflows is not relicensed under the MIT License. Each external tool remains subject to its own license and associated terms.

Users are responsible for consulting the respective software documentation and license information when installing or using third-party tools.

The software versions used for the analyses are documented above to support reproducibility. External software is not necessarily distributed with this repository.

Workflow Automation and HPC
Bash scripting is used for workflow automation, file management, and SLURM job submission.
Computationally intensive workflows are executed on a SLURM-managed high-performance computing (HPC) system. SLURM job arrays are used where appropriate to process independent samples or computational tasks in parallel.

Some Bash scripts act as wrappers around Python programs or external bioinformatics tools, while others contain the computational workflow and SLURM configuration directly.

Reproducibility
Before running the workflows, ensure that the required software versions, reference data, and computational resources are available.
The general setup process is:

Install or load Conda/mamba.
Create the required environments using the environment specification files provided in this directory.
Install or configure standalone external tools that are not included in the Conda environments.
Verify the relevant software versions before running the corresponding workflows.
Prepare the required reference data and input files according to the relevant workflow documentation.
Run the computational workflows using the Bash, Python, and SLURM scripts provided in the repository.
The exact commands, parameters, and software usage are documented in the relevant workflow scripts and directory-level README files.
Generated sequencing data, intermediate files, feature tables, model outputs, and analysis results are not stored in the repository.

The computational environment described here corresponds to the software configuration used for the analyses reported in the accompanying MSc thesis.