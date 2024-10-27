import os
import json
import re
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from util.utils import load_file_line_by_line
import logging
from threading import Lock
from tqdm import tqdm
import itertools

class combineFiles:
    def __init__(self, categories: dict, main_dict: dict, chunk_size: int, output_dir: str, max_workers = 10):
        self.categories = categories
        self.main_dict = main_dict
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.max_workers = max_workers

    def process_file(self, file_path, lock):
        count = 0 
        self.logger.info(f"Processed file: {file_path}")
        for item in load_file_line_by_line(file_path):
            corpusid = item['corpusid']
            if corpusid in self.main_dict:
                count+=1
                with lock:
                    self.main_dict[corpusid].update(item)
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
                for i, chunk in enumerate(chunks)
            ]
            # Ensure all files are written before exiting
            for future in futures:
                future.result()



category_names = ['abstracts', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 's2orc', 'tldrs']
base_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8'
output_dir = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/combined'

categories = {}

for category in category_names: 
    if category == 'abstracts':
        continue
    categories[category] = os.path.join(base_folder, category)

filterd_abstracts = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/abstract_filtered_more'
all_abstracts = {}

for dir in os.listdir(filterd_abstracts):
    file_path = os.path.join(filterd_abstracts, dir)
    with open(file_path, 'r') as file:
        data = json.load(file)
    temp_dict = {item['corpusid']: item for item in data}
    all_abstracts.update(temp_dict)

combiner = combineFiles(categories=categories, main_dict=all_abstracts, chunk_size=5000, output_dir=output_dir)
combiner.process_category()