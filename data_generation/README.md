Data generation
This directory contains the workflow used to generate the simulated MUC1 sequencing dataset used in the downstream analyses.
The data-generation workflow proceeds from preparation of the reference sequence through generation of positive and negative samples, read simulation, diploid sample construction, and final processing of the simulated reads.

Workflow
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
SHARK / read processing
      ↓
Processed reads
      ↓
k-mer feature generation / VNtyper2 analysis

The final processed reads are shared by both downstream analysis approaches. They are used as input to k-mer feature generation and are also analysed independently using VNtyper2.
Directory structure
reference_preparation/
Contains scripts used to identify and extract the relevant MUC1 reference sequence and associated annotation information from the reference genome.
The main steps include:

identifying the chromosome/contig containing MUC1;
extracting the relevant reference sequence;
extracting the corresponding GFF annotation;
converting annotation information into BED format where required.
sample_preparation/
Contains scripts used to prepare the positive and negative sequence material from which simulated samples are generated.
This includes mutation of the MUC1 sequence and conversion of annotation information required by downstream simulation steps.

The Bash wrappers use SLURM to run the underlying operations across multiple samples where appropriate.

read_simulation/
Contains the tools and scripts used to simulate and process sequencing reads.
The workflow includes:

preparation/indexing of reference FASTA files;
sequencing-read simulation;
filtering or processing of simulated reads;
generation of the processed reads used by downstream analyses.
The final read-processing step produces the sequencing data used by both the k-mer-based and VNtyper2 approaches.
sample_generation/
Contains the workflow used to construct the final diploid positive and negative samples from the prepared sequence material.
The positive and negative sample-generation scripts use SLURM arrays to process multiple samples.

submit_positive.sh and submit_negative.sh are used for SLURM job submission/management associated with the corresponding sample-generation workflows.

SLURM and execution
Many of the Bash scripts in this directory contain SLURM directives because the workflow is designed to process multiple samples on a computing cluster.
These scripts should be viewed primarily according to the computational step they perform rather than as generic automation scripts. For example, mutate_muc1.sh belongs to sample_preparation/ because it executes the mutation-generation workflow, even though it uses a SLURM array.

Output
The final output of the data-generation workflow is a collection of processed sequencing reads representing the simulated samples.
These processed reads form the input to the downstream stages:

Processed reads
      ↓
 ┌────┴───────────┐
 ↓                ↓
k-mer generation  VNtyper2
 ↓                ↓
ML analysis       baseline

The scripts for converting these processed reads into k-mer feature tables are located in ../feature_generation/.
The VNtyper2 analysis is located in ../analysis/baseline/.