from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType
import numpy as np
from typing import List, Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch

class SPECTER2(KVStore):
    def __init__(self, index_name: str, key_instruction: str, query_instruction: str, save_as_tensor: bool = False, model_path: str = "allenai/specter2_base"):
        super().__init__(index_name, 'specter2', save_as_tensor)

        #model config
        self.model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoAdapterModel.from_pretrained(self.model_path)
        self._model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        self._model.to("cuda")

        self.key_instruction = key_instruction
        self.query_instruction = query_instruction

        self.save_as_tensor = save_as_tensor

    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True, batch_size: int = 3) -> List[Any]:
        encoded_keys = []
        total_batches = utils.batch_size_calc(texts, batch_size)

        iterator = utils.batch_iterator(texts, batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, total=total_batches, desc="Encoding Batches")

        for text_batch in iterator:
            inputs = self._tokenizer(text_batch, 
                                        padding=True, 
                                        truncation=True,
                                        return_tensors="pt", 
                                        return_token_type_ids=False,
                                        max_length=512).to("cuda")
            output = self._model(**inputs)
            embeddings = output.last_hidden_state[:, 0, :].detach().clone().requires_grad_(False)
            encoded_keys.extend(embeddings)
        return encoded_keys
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        if self.save_as_tensor:
            inner_product = torch.matmul(self.encoded_keys, encoded_query)
            _, top_indices = torch.topk(inner_product, n)
            return top_indices

        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices
    
    def load(self, path: str):
        super().load(path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoAdapterModel.from_pretrained(self.model_path)
        self._model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        self._model.to("cuda")
        return self