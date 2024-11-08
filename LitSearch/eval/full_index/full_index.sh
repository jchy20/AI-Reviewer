#!/bin/bash
#SBATCH --job-name=full_index
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hc387@duke.edu

hostname && nvidia-smi && env
# Load the module for Anaconda/Miniconda if it's not automatically initialized
# Uncomment and modify the next line if necessary
# module load anaconda

source /home/users/hc387/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate litsearch
which python

# Navigate to the directory containing the Python script (if it's not in the home directory)
cd /usr/project/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index

python full_index.py
