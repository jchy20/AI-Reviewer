from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import os
import sys
sys.path.append(os.path.abspath("../../"))
from utils import utils
import json
import torch
import tqdm


class TitleCheck():
    def __init__(self, input_folder):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.input_folder = Path(input_folder)

    def process_each_file(self, file_path: Path):
        self.logger.info(f"Processing {file_path.name}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            try:
                utils.get_clean_corpusid(data[item])
            except Exception as e:
                self.logger.error('No corpusid')
            try:
                utils.get_clean_title_abstract(data[item])
            except Exception as e:
                self.logger.error('No title')
            try:
                data[item]['embedding_v2_vector']
            except Exception as e:
                self.logger.error('No embedding_v2_vector')

    def process_category(self):
        json_files = list(self.input_folder.glob('*.json'))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                    executor.submit(self.process_each_file, file_path): file_path
                    for file_path in json_files
                }

            for future in tqdm(as_completed(futures), total=len(json_files), desc='Processing all files'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f'{file_path} generated an exception: {exc}')



input_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/everything_combined_more'

title_check = TitleCheck(input_folder)
title_check.process_category()
