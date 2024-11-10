import os
import multiprocessing
import torch
from embed_filter import EmbeddingFilter
from combineFiles import combineFiles

if __name__ == '__main__':
    ###################### Load Seeds ######################
    seeds_path = '/usr/xtmp/hc387/ai_reviewer/data/iclr2024_retrieval_eval/iclr_seeds.pt'
    encoded_seeds = torch.load(seeds_path, map_location='cpu')
    ########################################################




    ############### Filter Embeddings ######################
    input_folder = '/usr/xtmp/hc387/ai_reviewer/data/semantic_scholar/2024_10_8/embeddings_v2'
    output_folder = '/usr/xtmp/hc387/ai_reviewer/LitSearch/eval/full_index/filtered_embeddings'

    multiprocessing.set_start_method('spawn', force=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NUM_GPUS'] = '4'

    knn_filter = EmbeddingFilter(encoded_seeds, input_folder, output_folder)
    knn_filter.process_all_files()
    ########################################################