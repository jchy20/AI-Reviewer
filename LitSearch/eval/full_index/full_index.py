import sys
import os
sys.path.append(os.path.abspath("../../"))
from utils import utils
import json
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from embed_filter import EmbeddingFilter
from combineFiles import combineFiles


#load embedding model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.to("cuda")


########################################################
############### Create KNN Seed ########################
iclr_path = '/usr/xtmp/hc387/ai_reviewer/data/iclr2024_retrieval_eval/iclr_2024_references_evaluated_abstract.json'
with open(iclr_path, 'r') as file:
    iclr_list = json.load(file)

embed_prep = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in iclr_list]
iterator = utils.batch_iterator(embed_prep[:20], 3)
encoded_seeds = []
for text_batch in iterator:
    inputs = tokenizer(text_batch, 
                                padding=True, 
                                truncation=True,
                                return_tensors="pt", 
                                return_token_type_ids=False,
                                max_length=512).to("cuda")
    output = model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :].detach().clone().requires_grad_(False)
    encoded_seeds.extend(embeddings)
encoded_seeds = torch.stack(encoded_seeds)

########################################################

########################################################
############### Filter Embeddings ######################
input_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/embeddings_v2'
output_folder = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/filtered_embeddings'
embed_filter = EmbeddingFilter(seeds = encoded_seeds, input_folder = input_folder, output_folder=output_folder)
embed_filter.process_all_files()
main_dict = embed_filter.return_dict()
embed_filter.write_all_chunks()
########################################################

category_names = {"abstracts": '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/abstracts', "papers": '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/papers'}

combiner = combineFiles(categories = category_names, main_dict = main_dict, chunk_size=100000, output_dir = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/combined_dict')
combiner.process_category()
combiner.write_all_chunks()

