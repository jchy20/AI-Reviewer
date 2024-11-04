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

class ManualClean():
    def __init__(self, input_folder: str, output_dir: str, model_path: str = "allenai/specter2_base"):
        # input output dir
        self._input_folder = Path(input_folder)
        self._output_dir = output_dir

        #specter v2 model config and load
        self.model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoAdapterModel.from_pretrained(self.model_path)
        self._model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        
        #index 
        self.index_name = 'specter2'
        self.index_type = 'specter2'
        self.key_instruction = "Represent the title and abstract of the research paper for retrieval:"
        self.query_instruction = "Represent the research question for retrieving relevant research paper abstracts:"
        self.keys = []
        self.values = []
        self.encoded_keys = []

        #others
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

    def process_each_file(self, file_path: Path, lock) -> None:
        #key is title and abstract
        #value is corpusid
        #embedding is torch.tensor
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            value = utils.get_clean_corpusid(data[item])

            #title and abstract
            abstract = utils.get_clean_abstract(data[item])
            try:
                title = utils.get_clean_title(data[item])
            except KeyError as e:
                title = ''
            key = f"Title: {title}\nAbstract: {abstract}"

            #embeddings
            try:
                embedding = ast.literal_eval(data[item]['embedding_v2_vector'])
                embedding = torch.tensor(embedding).requires_grad_(False)
                if len(embedding) != 768:
                    raise KeyError(f"embedding length is not 768")
            except KeyError as e:
                self._logger.error(f'No specter2 embedding for corpusid: {utils.get_clean_corpusid(data[item])}')

                papers = [{'title': title, 'abstract': abstract}]
                with lock:
                    text_batch = [d['title'] + self._tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
                    
                    # preprocess the input
                    inputs = self._tokenizer(text_batch, padding=True, truncation=True,
                                                    return_tensors="pt", return_token_type_ids=False, max_length=512)
                    output = self._model(**inputs)

                # take the first token in the batch as the embedding
                embeddings = output.last_hidden_state[:, 0, :]
                embedding =  embeddings[0].detach().clone().requires_grad_(False)
            
            with lock:
                self.keys.append(key)
                self.values.append(value)
                self.encoded_keys.append(embedding)
    
    def process_all_files(self) -> None:

        lock = Lock()
        
        json_files = list(self._input_folder.glob('*.json'))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                    executor.submit(self.process_each_file, file_path, lock): file_path
                    for file_path in json_files
                }

            for future in tqdm(as_completed(futures), total=len(json_files), desc='Processing all files'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self._logger.error(f'{file_path} generated an exception: {exc}')

    def return_vals(self, save_as_tensor: bool = False) -> None:
        if save_as_tensor:
            self.encoded_keys = torch.stack(self.encoded_keys)
        print(self.keys)
        print(self.values)
        print(self.encoded_keys)
    
    def save(self, save_as_tensor: bool = False) -> None:
        savetype = 'list'
        if save_as_tensor:
            self.encoded_keys = torch.stack(self.encoded_keys)
            savetype = 'tensor'
        save_dict = {}
        for key, value in self.__dict__.items():
            if key[0] != "_":
                save_dict[key] = value

        print(f"Saving index to {os.path.join(self._output_dir, f'{savetype}_{self.index_name}.{self.index_type}')}")
        os.makedirs(self._output_dir, exist_ok=True)
        with open(os.path.join(self._output_dir, f"{savetype}_{self.index_name}.{self.index_type}"), 'wb') as file:
            pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

input_dir = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/everything_combined_more'
output_dir = '/usr/project/xtmp/ai_reviewer/index'
temp = ManualClean(input_dir, output_dir)
temp.process_all_files()
temp.save()