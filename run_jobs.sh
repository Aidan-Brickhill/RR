#!/bin/bash

# Source the .bashrc file
source ~/.bashrc

# Activate the conda environment
conda activate handover_env

# Change to the specified directory
cd ~/research/RR/cluster

# Stash any changes in the git repository
git stash

# Pull the latest changes from the remote repository
git pull

# Submit the job multiple times using sbatch
sbatch job.batch

# Monitor the job queue
watch -n 1 squeue
