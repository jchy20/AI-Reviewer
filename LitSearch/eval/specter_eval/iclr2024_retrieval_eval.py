import sys
import os
import json
sys.path.append(os.path.abspath("../../"))
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.specter2 import SPECTER2
from transformers import AutoTokenizer
import torch

import numpy as np
from typing import List, Tuple, Dict, Set

class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Compute Precision@K
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Cutoff point
        """
        if k == 0 or len(retrieved_docs) == 0:
            return 0.0
            
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Compute Recall@K
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Cutoff point
        """
        if len(relevant_docs) == 0:
            return 0.0
            
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant_docs)
        return relevant_retrieved / len(relevant_docs)

    @staticmethod
    def mean_average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Compute Mean Average Precision
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
        """
        if len(relevant_docs) == 0:
            return 0.0

        precisions = []
        relevant_found = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                relevant_found += 1
                precisions.append(relevant_found / i)
                
        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """
        Compute NDCG@K (assuming binary relevance)
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Cutoff point
        """
        def dcg_at_k(rel_list: List[int], k: int) -> float:
            rel_list = rel_list[:k]
            return sum((2**rel - 1) / np.log2(pos + 2) for pos, rel in enumerate(rel_list))
        
        if k == 0 or len(retrieved_docs) == 0:
            return 0.0
            
        # Calculate actual DCG
        rel_list = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
        dcg = dcg_at_k(rel_list, k)
        
        # Calculate ideal DCG (sort relevant documents first)
        ideal_rel_list = sorted([1 if doc in relevant_docs else 0 for doc in retrieved_docs], reverse=True)
        idcg = dcg_at_k(ideal_rel_list, k)
        
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """
        Compute Mean Reciprocal Rank
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0

    @staticmethod
    def compute_coverage_stats(data: List[Dict], all_corpus_ids: Set[str]) -> Dict[str, float]:
        """
        Compute coverage statistics for queries and their references
        Args:
            data: List of query documents with their references
            all_corpus_ids: Set of all document IDs in the retrieval system
        Returns:
            Dictionary containing coverage statistics
        """
        total_queries = len(data)
        found_queries = 0
        ref_coverage_per_query = []
        total_unique_refs = set()
        found_unique_refs = set()

        for doc in data:
            # Check if query document exists in index
            if int(doc['corpusId']) in all_corpus_ids:
                found_queries += 1
            
            # Check reference coverage
            doc_refs = {int(ref['corpusId']) for ref in doc['references'] if ref['corpusId'] is not None}
            total_unique_refs.update(doc_refs)
            found_refs = doc_refs.intersection(all_corpus_ids)
            found_unique_refs.update(found_refs)
            
            # Calculate reference coverage for this query
            ref_coverage = len(found_refs) / len(doc_refs) if doc_refs else 0
            ref_coverage_per_query.append(ref_coverage)

        return {
            'query_coverage': found_queries / total_queries if total_queries > 0 else 0,
            'avg_reference_coverage': np.mean(ref_coverage_per_query),
            'total_unique_refs': len(total_unique_refs),
            'found_unique_refs': len(found_unique_refs),
            'global_reference_coverage': len(found_unique_refs) / len(total_unique_refs) if total_unique_refs else 0
        }
parser = argparse.ArgumentParser()
parser.add_argument("--topk", required=True, type=int) 
args = parser.parse_args()

# 1. load model
# specter2_tensor = None
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base",trust_remote_code=True)
index_path = '/usr/project/xtmp/ai_reviewer/index/list_specter2.specter2'
specter2_tensor = SPECTER2(None, None, None)
specter2_tensor = specter2_tensor.load(index_path)
specter2_tensor.save_as_tensor = True
specter2_tensor.encoded_keys = torch.stack(specter2_tensor.encoded_keys).to("cuda")


# 2. load data
path = "/home/users/jw834/projects/ai_reveiwer/AI-Reviewer/data/iclr2024_retrieval_eval/iclr_2024_references_evaluated_abstract.json"
with open(path, "r") as f:
    data = json.load(f)
print(data[0])

def evaluate_retrieval(data: List[Dict], specter2_tensor, tokenizer, max_k: int):
    # Define K values
    k_values = [1, 3, 5, 10, 20, 30, 40, 50]
    k_values = [k for k in k_values if k <= max_k]  # Only use K values up to max_k

    all_corpus_ids = set(specter2_tensor.values)  # Adjust this based on your KVStore structure
    print(f"Total documents in index: {len(all_corpus_ids)}, type: {type(specter2_tensor.values[0])}")
    coverage_stats = RetrievalMetrics.compute_coverage_stats(data, all_corpus_ids)
    
    # Initialize metrics dictionary
    metrics = {
        'p@k': {k: [] for k in k_values},
        'r@k': {k: [] for k in k_values},
        'ndcg@k': {k: [] for k in k_values},
        'map': [],
        'mrr': []
    }
    
    for d in tqdm(data):
        title = d["title"]
        abstract = d["abstract"]
        if abstract is None:
            abstract = ""
        query = title + tokenizer.sep_token + abstract
        
        # Convert references to set of corpusIds for easier lookup
        relevant_docs = {ref['corpusId'] for ref in d['references']}
        
        # Get retrieved documents
        retrieved_results = specter2_tensor.query(query, max_k, True)
        retrieved_docs = [str(doc[1]) for doc in retrieved_results]  # Assuming doc[1] is corpusId

        
        # Calculate metrics for each K
        for k in k_values:
            metrics['p@k'][k].append(RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, k))
            metrics['r@k'][k].append(RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, k))
            metrics['ndcg@k'][k].append(RetrievalMetrics.ndcg_at_k(retrieved_docs, relevant_docs, k))
        
        # Calculate K-independent metrics
        metrics['map'].append(RetrievalMetrics.mean_average_precision(retrieved_docs, relevant_docs))
        metrics['mrr'].append(RetrievalMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs))
    
    # Calculate means
    results = {
        'map': np.mean(metrics['map']),
        'mrr': np.mean(metrics['mrr'])
    }
    
    # Add K-dependent metrics
    for k in k_values:
        results[f'p@{k}'] = np.mean(metrics['p@k'][k])
        results[f'r@{k}'] = np.mean(metrics['r@k'][k])
        results[f'ndcg@{k}'] = np.mean(metrics['ndcg@k'][k])
    
    results.update(coverage_stats)
    return results

results = evaluate_retrieval(data, specter2_tensor, tokenizer, args.topk)



# Print results
print("\nCoverage Statistics:")
print(f"Query Document Coverage: {results['query_coverage']:.1%}")
print(f"Average Reference Coverage per Query: {results['avg_reference_coverage']:.1%}")
print(f"Global Reference Coverage: {results['global_reference_coverage']:.1%}")
print(f"Total Unique References: {results['total_unique_refs']}")
print(f"Found Unique References: {results['found_unique_refs']}")



print("\nRetrieval Evaluation Results:")
print(f"MAP: {results['map']:.3f}")
print(f"MRR: {results['mrr']:.3f}")

# Print K-dependent metrics
k_values = [1, 3, 5, 10, 20, 30, 40, 50]
k_values = [k for k in k_values if k <= args.topk]

print("\nPrecision@K:")
for k in k_values:
    print(f"P@{k}: {results[f'p@{k}']:.3f}")

print("\nRecall@K:")
for k in k_values:
    print(f"R@{k}: {results[f'r@{k}']:.3f}")

print("\nNDCG@K:")
for k in k_values:
    print(f"NDCG@{k}: {results[f'ndcg@{k}']:.3f}")