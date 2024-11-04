from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import sys
import os
sys.path.append(os.path.abspath("../../"))
from utils import utils
from eval.retrieval.specter2 import SPECTER2
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT', 'corpusid': 123},
          {'title': 'Attention is all you need', 'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks', 'corpusid': 2342354},
          {'title': 'Situated Faithfulness', 'abstract': 'In this work, we explore whether LLM is faithful to the promot', 'corpusid': 1029310938}]


parser = argparse.ArgumentParser()
parser.add_argument("--index_type", required=True) # bm25, instructor, e5, gtr, grit
parser.add_argument("--key", required=True) # title_absract, full_paper, paragraphs
parser.add_argument("--save_as_tensor", required=True) # True or False 
args = parser.parse_args()


if args.key == "title[SEP]abstract":
    query_instruction = "Represent the research question for retrieving relevant research paper abstracts:"
    key_instruction = "Represent the title and abstract of the research paper for retrieval:"
else:
    raise ValueError("Invalid key")
if args.save_as_tensor == 'True':
    save_as_tensor = True
elif args.save_as_tensor == 'False':
    save_as_tensor = False
else:
    raise ValueError("Invalid key")

specter2 = SPECTER2(args.index_type, key_instruction, query_instruction, save_as_tensor)
kv_pairs = {utils.get_title_abstract_SEPtoken(record): utils.get_clean_corpusid(record) for record in papers}


specter2.create_index(kv_pairs)

specter2.save('/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/temp_testing/temp_pickle')