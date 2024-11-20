import sys
import os
sys.path.append(os.path.abspath("../../"))
from utils.utils import load_file_line_by_line
from pathlib import Path
from threading import Lock
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

class FindFilter():
    def __init__(self, input_folder: str, output_path: str, topk: int):
        self.main_dict = {}
        self.input_folder = Path(input_folder)
        self.output_path = output_path
        self.topk = topk

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.max_workers = 10

    def process_single_file(self, file_path, lock):
        try:
            for item in load_file_line_by_line(file_path):
                corpusid = item['corpusid']
                with lock:
                    self.main_dict[corpusid] = item['avg_similarity']
        except Exception as e:
            self.logger.error(f"{e}")

    def process_all_file(self):
        lock = Lock()
        all_files = list(self.input_folder.glob('*.jsonl'))

        total_files = len(all_files)
        self.logger.info(f"Total files to combine: {total_files}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files to the executor
            futures = {
                executor.submit(self.process_single_file, file_path, lock): file_path
                for file_path in all_files
            }

            # Use tqdm to track the progress of all futures
            for future in tqdm(as_completed(futures), total=total_files, desc='Merging Main'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f'{file_path} generated an exception: {exc}')
    
    def sort_and_write(self):
        top_k_dict = dict(sorted(self.main_dict.items(), key=lambda item: item[1], reverse=True)[:self.topk])
        with open(self.output_path, 'w') as json_file:
            json.dump(top_k_dict, json_file, indent=4)

input_folder = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/filtered_embeddings'
output_file = 'sorted_k_corpusid.json'

topkSort = FindFilter(input_folder, output_file, 5000000)

topkSort.process_all_file()
topkSort.sort_and_write()