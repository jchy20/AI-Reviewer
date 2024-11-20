import os
import sys
sys.path.append(os.path.abspath("../../"))
from utils import utils
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import re
from threading import Lock


class CombineFiles:
    def __init__(self, main_dict: str, embedding_folder: str, paper_folder: str, abstract_folder: str, paperid_folder: str, output_folder: str):
        #input folders
        self.embedding_folder = Path(embedding_folder)
        self.paper_folder = Path(paper_folder)
        self.abstract_folder = Path(abstract_folder)
        self.paperid_folder = Path(paperid_folder)
        self.output_folder = Path(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        
        #load kv pairs for corpusid, avg_sim
        with open(main_dict, 'r') as f:
            self.main_dict = json.load(f)

        #others
        self.max_workers = 10
        self.chunk_size = 50000
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def merge_main(self, file_path, lock):
        for item in utils.load_file_line_by_line(file_path):
            try:
                corpusid = item['corpusid']

                if str(corpusid) in self.main_dict:
                    
                    update_dict = {}
                    #template for updating embeddings
                    if re.search('embeddings', str(file_path), re.IGNORECASE):
                        update_dict = {"embeddings": item['embeddings'], "avg_similarity": item['avg_similarity']}
                    #template for updating title and publicationdate
                    elif re.search('papers', str(file_path), re.IGNORECASE):
                        update_dict = {"title": item['title'], "publicationdate": item['publicationdate']}
                    #template for updating abstract
                    elif re.search('abstracts', str(file_path), re.IGNORECASE):
                        update_dict = {"abstract": item['abstract']}
                    #template for updating paper_id
                    elif re.search('paper_ids', str(file_path), re.IGNORECASE):
                        update_dict = {"paper_ids": item['sha']}
                    else:
                        self.logger.error(f"Error creating an update dict for {corpusid}")
                
                        
                    if isinstance(self.main_dict[str(corpusid)], dict):
                        with lock:
                            self.main_dict[str(corpusid)].update(update_dict)
                    elif isinstance(self.main_dict[str(corpusid)], float):
                        with lock:
                            self.main_dict[str(corpusid)] = update_dict
                    else:
                        self.logger.error(f"main_dict has update for {corpusid} not successful")

            except Exception as exc:
                self.logger.error(f"Error in file {file_path}: {exc}", exc_info=True)

    def merge_main_all(self):
        lock = Lock()
        all_files = list(self.embedding_folder.glob('*')) + list(self.paper_folder.glob('*')) + list(self.abstract_folder.glob('*')) + list(self.paperid_folder.glob('*'))

        total_files = len(all_files)
        self.logger.info(f"Total files to combine: {total_files}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files to the executor
            futures = {
                executor.submit(self.merge_main, file_path, lock): file_path
                for file_path in all_files
            }

            # Use tqdm to track the progress of all futures
            for future in tqdm(as_completed(futures), total=total_files, desc='Merging Main'):
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
                executor.submit(self.write_chunk_to_file, chunk, f"{self.output_folder}/file_{i + 1}.json")
                for i, chunk in enumerate(chunks)
            ]
            # Ensure all files are written before exiting
            for future in futures:
                future.result()


abs = '/usr/xtmp/hc387/ai_reviewer/data/s2/2024-11-12/abstracts'
emb = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/filtered_embeddings'
pap = '/usr/xtmp/hc387/ai_reviewer/data/s2/2024-11-12/papers'
papid = '/usr/xtmp/hc387/ai_reviewer/data/s2/2024-11-12/paper_ids'
out = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/ready2index'
main_dict_path = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/sorted_k_corpusid.json'

combine = CombineFiles(main_dict_path, emb, pap, abs, papid, out)

combine.merge_main_all()
combine.write_all_chunks()