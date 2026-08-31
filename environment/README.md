# Computational Environment

This directory contains environment specifications and documentation describing the software and computational requirements used in the project.

The computational workflows were developed and primarily executed in a Linux-based, SLURM-managed high-performance computing (HPC) environment. The environments and software specifications are provided to support local reproduction where feasible.

## Programming Languages and Environment Managers

The project uses the following programming languages and software environments:

| Software | Version | Purpose |
|---|---|---|
| Python | 3.8.20 | NEAT 3.4 compatibility and custom FASTQ/GFF/GFF3 processing scripts |
| Python | 3.11.13 | Main machine-learning workflows |
| Python | 3.11.15 | Domain-adversarial neural network (DANN) implementation |
| Python | 3.13.7 | VNtyper (2.0) baseline analysis |
| R | 4.3.2 | Statistical analysis and data visualisation |
| mamba | — | Alternative package manager for creating the Sourmash environment |

## Environment Overview

| Environment | Manager | Python version | Main software |
|---|---|---|---|
| neat34_env | Conda | 3.8.20 | NEAT 3.4 |
| ml_env | Conda | 3.11.13 | Machine-learning workflows |
| dann_env | Conda | 3.11.15 | DANN implementation |
| vntyper | Python venv | 3.13.7 | VNtyper 2.0 |
| smash | mamba | — | Sourmash |

Dedicated environment specifications are provided in this directory where applicable.

Available environment files:

| File | Environment | Purpose |
|---|---|---|
| ml_env.yml | ml_env | Main machine-learning workflows |
| dann_env.yml | dann_env | Domain-adversarial neural network workflow |

NEAT-3.4 is installed from source rather than through a Conda environment file; installation instructions are provided below.

## Environment Setup

The environment specification files provided in this directory can be used to recreate the computational environments used for the project.

### Conda

If Conda is available, an environment can generally be created from an environment specification (ml_env.yml, dann_env.yml) using:

```bash
conda env create -f <environment_file>.yml
```

The environment can then be activated using:

```bash
conda activate <environment_name>
```

The requirements for creating a NEAT-3.4 conda environment are:

* python >= 3.8
* biopython == 1.79
* matplotlib >= 3.3.4 (optional, for plotting utilities)
* matplotlib-venn >= 0.11.6 (optional, for plotting utilities)
* pandas >= 1.2.1
* numpy >= 1.22.2
* pysam >= 0.16.0.1

1. Download NEAT-3.4 from:
   https://github.com/ncsa/NEAT/releases

2. Create a conda environment

```bash
conda create -n neat34_env python=3.8
```
3. Activate the environment and install NEAT-3.4 into it 

```bash
conda activate neat34_env
python -m pip install -e /path/to/NEAT-3.4
```
Replace `/path/to/NEAT-3.4` with the location of the extracted NEAT source directory. Additional packages should only be installed if missing.

### Mamba

The Sourmash environment managed using mamba can be created following the instructions at: https://sourmash.readthedocs.io/en/latest/tutorial-install.html

To install sourmash, create a new environment named smash and install sourmash:

```bash
mamba create -y -n smash sourmash-minimal
```

The environment can then be activated using:

```bash
mamba activate smash
```

### Python venv environment

VNtyper 2.0 can be installed using Python's built-in `venv` module. The following example shows installation on a SLURM-managed HPC system:

On local systems, omit the `module` commands and ensure that Python 3.13.7 is available before creating the environment.

```bash
cd $HOME
module purge
module load python/3.13.7
python -m venv vntyper

source vntyper/bin/activate

git clone https://github.com/hassansaei/vntyper.git
cd vntyper
pip install .
pip install matplotlib plotly jinja2 requests
```

Then, to use the application:

```bash
module purge
module load python/3.13.7
source $HOME/vntyper/bin/activate
```

Alternative installation instructions for VNtyper 2.0 can be found at (https://github.com/maxkrou/VNtyper#installation)

After creating an environment, verify the relevant software versions before running the corresponding workflow.

For example:

```bash
python --version

```
Some software dependencies are installed separately from the managed environments. These tools must be installed and configured independently before running workflows that require them.

### Bioinformatics Software
The computational workflows use a combination of environment-managed software and standalone bioinformatics tools.

The principal tools and versions used in the project are:

| Tool | Version | Purpose |
|---|---|---|
| NEAT | 3.4 | Sequencing-read simulation |
| DWGSIM | 0.1.17-dev | Sequencing-read simulation |
| seqtk | 1.5 | Sequence processing |
| Shark | 1.2.0 | MUC1 read extraction |
| KMC | 3.2.4 | k-mer counting |
| BWA | 0.7.18 | Sequence alignment |
| SAMtools | 1.23.1 | BAM/SAM processing |
| MUMmer | 4.0.0beta2 | Genome alignment |
| minimap2 | 2.28 | Sequence alignment |
| Sourmash | 4.9.4 | Sequence similarity and sketching |
| VNtyper | 2.0 (Kestrel) | MUC1 baseline analysis |

Some tools are managed through dedicated Conda or mamba environments, while others are installed separately and invoked directly by the workflow scripts.

### External Software and Licensing
The MIT License in the root of this repository applies to the original code and workflow scripts distributed in this repository.
Third-party software used by the workflows is not relicensed under the MIT License. Each external tool remains subject to its own license and associated terms.

Users are responsible for consulting the respective software documentation and license information when installing or using third-party tools.

The software versions used for the analyses are documented above to support reproducibility. External software is not necessarily distributed with this repository.

### Workflow Automation and HPC
Bash scripting is used for workflow automation, file management, and SLURM job submission.

Computationally intensive workflows are executed on a SLURM-managed high-performance computing (HPC) system. SLURM job arrays are used where appropriate to process independent samples or computational tasks in parallel.

Some Bash scripts act as wrappers around Python programs or external bioinformatics tools, while others contain the computational workflow and SLURM configuration directly.

### Reproducibility
Before running the workflows, ensure that the required software versions, reference data, and computational resources are available.

The general setup process is:

1. Install or load Conda/mamba.
2. Create the required environments using the environment specification files provided in this directory.
3. Install or configure standalone external tools that are not included in the Conda environments.
4. Verify the relevant software versions before running the corresponding workflows.
5. Prepare the required reference data and input files according to the relevant workflow documentation.
6. Run the computational workflows using the Bash, Python, and SLURM scripts provided in the repository.


The exact commands, parameters, and software usage are documented in the relevant workflow scripts and directory-level README files.

Generated sequencing data, intermediate files, feature tables, model outputs, and analysis results are not stored in the repository.

The computational environment described here corresponds to the software configuration used for the analyses reported in the accompanying MSc thesis. Later software versions may require additional testing before being used to reproduce the results.