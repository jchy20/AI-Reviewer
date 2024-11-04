import sys
import os
sys.path.append(os.path.abspath("../../"))
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.specter2 import SPECTER2
from transformers import AutoTokenizer
import torch

self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
index_path = '/usr/project/xtmp/ai_reviewer/index/list_specter2.specter2'
specter2_tensor = SPECTER2(None, None, None)
specter2_tensor = specter2_tensor.load(index_path)
specter2_tensor.save_as_tensor = True
specter2_tensor.encoded_keys = torch.stack(specter2_tensor.encoded_keys).to("cuda")


parser = argparse.ArgumentParser()
parser.add_argument("--topk", required=True, type=int) 

query = [{"title": "A new method for image recognition", "abstract": "We propose a new method for image recognition that achieves state-of-the-art performance on the ImageNet dataset."}]

text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in query]
print(text_batch)
topk = specter2_tensor.query(text_batch, args.topk, True)
print(topk)