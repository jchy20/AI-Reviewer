#!/bin/bash
#SBATCH --job-name=list
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
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
cd /usr/xtmp/hc387/ai_reviewer/LitSearch/eval/temp_testing

# Execute the Python script

#returned results
TOPK=20 
#query
QUERY="Enhancing Large Language Models' Situated Faithfulness to External Contexts[SEP]Large Language Models (LLMs) are often augmented with external information as contexts, but this external information can sometimes be inaccurate or even intentionally misleading. We argue that robust LLMs should demonstrate situated faithfulness, dynamically calibrating their trust in external information based on their confidence in the internal knowledge and the external context. To benchmark this capability, we evaluate LLMs across several QA datasets, including a newly created dataset called RedditQA featuring in-the-wild incorrect contexts sourced from Reddit posts. We show that when provided with both correct and incorrect contexts, both open-source and proprietary models tend to overly rely on external information, regardless of its factual accuracy. To enhance situated faithfulness, we propose two approaches: Self-Guided Confidence Reasoning (SCR) and Rule-Based Confidence Reasoning (RCR). SCR enables models to self-access the confidence of external information relative to their own internal knowledge to produce the most accurate answer. RCR, in contrast, extracts explicit confidence signals from the LLM and determines the final answer using predefined rules. Our results show that for LLMs with strong reasoning capabilities, such as GPT-4o and GPT-4o mini, SCR outperforms RCR, achieving improvements of up to 24.2% over a direct input augmentation baseline. Conversely, for a smaller model like Llama-3-8B, RCR outperforms SCR. Fine-tuning SCR with our proposed Confidence Reasoning Direct Preference Optimization (CR-DPO) method improves performance on both seen and unseen datasets, yielding an average improvement of 8.9% on Llama-3-8B. In addition to quantitative results, we offer insights into the relative strengths of SCR and RCR. Our findings highlight promising avenues for improving situated faithfulness in LLMs. The data and code are released." #query

python temp_test.py --topk "$TOPK" --query "$QUERY"
