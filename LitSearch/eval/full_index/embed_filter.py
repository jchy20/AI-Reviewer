import sys
import os
sys.path.append(os.path.abspath("../../"))
from utils import utils
import json
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import ast
from threading import Lock
import pickle
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import torch.nn.functional as F
from itertools import islice
import heapq
import re

def load_file_line_by_line(file_path):
    '''
    return an iterable file
    '''
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

class EmbeddingFilter:
    def __init__(self, seeds: torch.Tensor, input_folder: str, output_folder: str, num_papers: int = 5000000, max_workers: int = 10):
        self.seeds = seeds.to("cuda")
        self.input_folder = Path(input_folder)
        self.num_papers = num_papers
        self.main_dict = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.max_workers = max_workers
        self.chunk_size = 100000

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_file(self, file_path: Path, lock) -> None:
        for data in load_file_line_by_line(file_path):
            try:
                embedding = ast.literal_eval(data['vector'])
                embedding = torch.tensor(embedding).float().requires_grad_(False).to("cuda")

                if len(embedding) != 768:
                    raise TypeError(f"embedding length is not 768")
                
                inner_product = F.cosine_similarity(self.seeds, embedding)
                avg_similarity = torch.mean(inner_product).item()

                embedding = embedding.cpu().tolist()
                with lock:
                    self.main_dict[data['corpusid']] = {"corpusid": data['corpusid'], "embeddings": embedding, "avg_similarity": avg_similarity}

            except KeyError as e:
                self.logger.error(f"No specter2 embedding for corpusid: {data['corpusid']}") 

            except TypeError as e:
                self.logger.error(e)
    
    def process_all_files(self) -> None:

        lock = Lock()
        
        json_files = list(self.input_folder.glob('*.json'))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                    executor.submit(self.process_file, file_path, lock): file_path
                    for file_path in json_files
                }

            for future in tqdm(as_completed(futures), total=len(json_files), desc='Processing all files'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f"{file_path} generated an exception: {exc}")

    def create_chunks(self, slice_keys):
        """Creates chunks from a slice of dictionary keys."""
        chunks = []
        chunk = {}
        for key in slice_keys:
            chunk[key] = self.main_dict[key]
            if len(chunk) >= self.chunk_size:
                chunks.append(chunk)
                chunk = {}
        if chunk:  # Add remaining items
            chunks.append(chunk)
        return chunks

    def parallel_chunking(self):
        # Step 1: Split keys into approximately equal slices
        keys = list(self.main_dict.keys())
        slice_size = max(1, len(keys) // self.max_workers)
        key_slices = [keys[i:i + slice_size] for i in range(0, len(keys), slice_size)]
        
        # Step 2: Use ThreadPoolExecutor to create chunks in parallel
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.create_chunks, slice_keys) for slice_keys in key_slices]
            
            # Collect results from each future
            for future in futures:
                all_chunks.extend(future.result())
        
        return all_chunks

    def write_all_chunks(self):
        """Combines chunking and file writing in parallel."""
        # Generate all chunks using parallel chunking
        chunks = self.parallel_chunking()
        
        # Write each chunk to a separate file in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.write_chunk_to_file, chunk, f"{self.output_folder}/file_{i + 1}.json")
                for i, chunk in enumerate(chunks) if chunk
            ]
            # Ensure all files are written before exiting
            for future in futures:
                future.result()

    def return_dict(self):
        top_n_items = dict(heapq.nlargest(self.num_papers, self.main_dict.items(), key=lambda item: item[1]["avg_similarity"]))
        self.main_dict = top_n_items
        
        return top_n_items