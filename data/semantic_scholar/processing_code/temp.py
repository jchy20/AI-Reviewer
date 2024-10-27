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

class combineFiles:
    def __init__(self, categories: dict, main_dict: dict):
        self.categories = categories
        self.main_dict = main_dict
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_file(self, file_path, lock):
        self.logger.info(f"Processed file: {file_path}")
        for item in load_file_line_by_line(file_path):
            corpusid = item['corpusid']
            if corpusid in self.main_dict:
                with lock:
                    self.main_dict[corpusid].update(item)
        self.logger.info(f"Finished processing: {file_path}")

    def process_category(self, max_workers):
        """Process all JSON files within all category folders concurrently."""
        lock = Lock()
        all_files = []

        # Collect all JSON files from all categories
        for category in self.categories:
            json_files = list(Path(self.categories[category]).glob('*.json'))
            all_files.extend(json_files)

        total_files = len(all_files)
        self.logger.info(f"Total files to process: {total_files}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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



category_names = ['abstracts', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 's2orc', 'tldrs']
base_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8'
# category_names = ['abstracts', 'authors', 'citations', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 'publication_venues', 's2orc', 'tldrs']

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

combiner = combineFiles(categories, all_abstracts)
combiner.process_category(max_workers=20)