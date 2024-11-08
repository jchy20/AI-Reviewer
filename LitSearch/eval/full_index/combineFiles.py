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


class combineFiles:
    def __init__(self, categories: dict, main_dict: dict, chunk_size: int, output_dir: str, max_workers = 10):
        self.categories = categories
        self.main_dict = main_dict

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_workers = max_workers

    def process_file(self, file_path, lock):
        count = 0 
        self.logger.info(f"Processing file: {file_path}")
        for item in load_file_line_by_line(file_path):
            corpusid = item['corpusid']

            if corpusid in self.main_dict:
                count+=1
                if re.search('abstracts', str(file_path), re.IGNORECASE):
                    with lock:
                        self.main_dict[corpusid].update({'abstract': item['abstract']})
                if re.search('papers', str(file_path), re.IGNORECASE):
                    with lock:
                        self.main_dict[corpusid].update({'title': item['title'], 'year': item['year']})
        self.logger.info(f"Finished processing: {file_path.name}. Found {count} matching records")

    def process_category(self):
        """Process all JSON files within all category folders concurrently."""
        lock = Lock()
        all_files = []

        # Collect all JSON files from all categories
        for category in self.categories:
            json_files = list(Path(self.categories[category]).glob('*.json'))
            all_files.extend(json_files)

        total_files = len(all_files)
        self.logger.info(f"Total files to process: {total_files}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files to the executor
            futures = {
                executor.submit(self.process_file, file_path, lock): file_path
                for file_path in all_files
            }

            # Use tqdm to track the progress of all futures
            for future in tqdm(as_completed(futures), total=total_files, desc='Processing all files'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f'{file_path} generated an exception: {exc}')

    def write_chunk_to_file(self, chunk, file_path):
        with open(file_path, 'w') as f:
            json.dump(chunk, f)

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
                executor.submit(self.write_chunk_to_file, chunk, f"{self.output_dir}/file_{i + 1}.json")
                for i, chunk in enumerate(chunks) if chunk
            ]
            # Ensure all files are written before exiting
            for future in futures:
                future.result()

    def return_dict(self):
        return self.main_dict