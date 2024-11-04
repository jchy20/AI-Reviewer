import sys
import os
sys.path.append(os.path.abspath("../../"))
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.specter2 import SPECTER2
import torch

index_path = '/usr/project/xtmp/ai_reviewer/index/list_specter2.specter2'
specter2_tensor = SPECTER2(None, None, None)
specter2_tensor = specter2_tensor.load(index_path)
specter2_tensor.save_as_tensor = True
specter2_tensor.encoded_keys = torch.stack(specter2_tensor.encoded_keys)
print(type(specter2_tensor.encoded_keys))


parser = argparse.ArgumentParser()
parser.add_argument("--topk", required=True, type=int) 
parser.add_argument("--query", required=False, type=str)
args = parser.parse_args()


topk = specter2_tensor.query(args.query, args.topk, True)
print(topk)