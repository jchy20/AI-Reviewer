#!/bin/bash
#SBATCH -A bhuwan
#SBATCH --partition=bhuwan
#SBATCH --nodes=1
#SBATCH --mem=40g
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=1-1:1
#SBATCH --job-name=alpaca
#SBATCH --output=eval%a.log

python iclr2024_retrieval_eval.py --topk 100 