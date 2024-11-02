import os
import json
import re
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from threading import Lock
from tqdm import tqdm
import itertools

def load_file_line_by_line(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())  # Ensure to strip any whitespace including newline


class JSONFilter:
    def __init__(self, titles_file: str, input_directory: str):
        self.titles_file = titles_file
        self.input_dir = Path(input_directory)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.titles_set = self.load_titles()

        # Initialize a lock for thread-safe operations
        self.lock = threading.Lock()

        self.matched_records = {}

    def load_titles(self):
        """Load titles from the text file into a set."""
        titles = set()
        with open(self.titles_file, 'r', encoding='utf-8') as f:
            for line in f:
                title = line.strip()
                if title:
                    titles.add(title)
        return titles
        
    def process_file(self, json_file: Path) -> None:
        """Process a single JSON file, matching records based on the title."""
        try:
            counts = 0
            print(f'Processing {json_file.name} now')
            local_matched_records = []
            for data in load_file_line_by_line(json_file):
                if 'title' in data:
                    # Process the title from the JSON record
                    processed_title = data['title'].lower()
                    if processed_title in self.titles_set:
                        # Extract corpusid and original title
                        with self.lock:
                            self.matched_records[data['corpusid']] = {
                                "corpusid": data.get('corpusid'),
                                "title": data.get('title')
                            }
                        counts += 1
                else:
                    self.logger.warning(f"No 'title' in record in file {json_file.name}")

            print(f'For {json_file.name}, found {counts} matching records')
            self.logger.info(f"Processed and matched: {json_file.name}")

        except Exception as e:
            self.logger.error(f"Error processing {json_file.name}: {str(e)}")

    def process_all_files(self, max_workers: int = None) -> None:
        """
        Process all JSON files in parallel using a thread pool.

        Args:
            max_workers: Maximum number of threads to use
        """
        json_files = list(self.input_dir.glob('*.json'))

        if not json_files:
            self.logger.warning(f"No JSON files found in {self.input_dir}")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_file, json_files)

    def return_found(self):
        return self.matched_records
    

class combineFiles:
    def __init__(self, categories: dict, main_dict: dict, output_dir: str, max_workers = 10):
        self.categories = categories
        self.main_dict = main_dict
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.max_workers = max_workers

    def process_file(self, file_path, lock):
        count = 0 
        self.logger.info(f"Processed file: {file_path}")
        for item in load_file_line_by_line(file_path):
            corpusid = item['corpusid']

            if corpusid in self.main_dict:
                count+=1

                if re.search('embeddings_v1', str(file_path), re.IGNORECASE):
                    key_mapping = {"model": "embedding_v1", "vector": "embedding_v1_vector"}
                    for old_key, new_key in key_mapping.items():
                        if old_key in item:
                            item[new_key] = item.pop(old_key)

                if re.search('embeddings_v2', str(file_path), re.IGNORECASE):
                    key_mapping = {"model": "embedding_v2", "vector": "embedding_v2_vector"}
                    for old_key, new_key in key_mapping.items():
                        if old_key in item:
                            item[new_key] = item.pop(old_key)
                
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

    def write_main_dict_to_file(self):
        with open(self.output_dir, 'w', encoding='utf-8') as f:
            json.dump(self.main_dict, f, indent=2)

title_file = '/usr/xtmp/hc387/ai_reviewer/data/iclr/titles.txt'
input_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/papers'
output_dir = '/usr/xtmp/hc387/ai_reviewer/data/iclr/iclr500.json'

categories = {
    "embedding_v1": '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/embeddings_v1',
    "embedding_v2": '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/embeddings_v2'
}

jsonfilter = JSONFilter(title_file, input_folder)
jsonfilter.process_all_files(10)
found = jsonfilter.return_found()

combiner = combineFiles(categories, found, output_dir)
combiner.process_category()
combiner.write_main_dict_to_file()