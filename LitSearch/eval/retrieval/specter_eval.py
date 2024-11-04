import os
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.specter2 import SPECTER2

index_path = '/usr/project/xtmp/ai_reviewer/index/tensor_specter2.specter2'
specter2_tensor = SPECTER2(None, None, None)
specter2_tensor = specter2_tensor.load(index_path)
specter2_tensor.save_as_tensor = True


parser = argparse.ArgumentParser()

parser.add_argument("--top_k", type=int, required=False, default=200)
parser.add_argument("__query", type=str, required=True)
args = parser.parse_args()


for query in tqdm(query_set):
    query_text = query["query"]
    top_k = index.query(query_text, args.top_k)
    query["retrieved"] = top_k

os.makedirs(args.retrieval_results_root_dir, exist_ok=True)
output_path = os.path.join(args.retrieval_results_root_dir, f"{args.index_name}.jsonl")
utils.write_json(query_set, output_path)
