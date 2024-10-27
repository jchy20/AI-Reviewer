#!/bin/bash
#SBATCH --job-name=jamescai_filter
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hc387@duke.edu

hostname && nvidia-smi && env
# Load the module for Anaconda/Miniconda if it's not automatically initialized
# Uncomment and modify the next line if necessary
# module load anaconda

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate review
which python

# Navigate to the directory containing the Python script (if it's not in the home directory)
cd /usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/processing_code

# Execute the Python script
python temp.py
