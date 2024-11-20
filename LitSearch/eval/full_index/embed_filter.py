import os
import sys
sys.path.append(os.path.abspath("../../"))
from utils.utils import load_file_line_by_line
import ast
import torch
import torch.nn.functional as F
import json
import logging
import gc
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import time

class EmbeddingFilter:
    def __init__(self, seeds: torch.Tensor, input_folder: str, output_folder: str):
        # Do not normalize seeds here or perform any CUDA operations
        self.seeds = seeds  # Store seeds as-is
        self.input_folder = Path(input_folder)
        self.output_folder = output_folder
        self.main_dict = {}

        os.makedirs(self.output_folder, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_file(self, file_path: Path, device_id: int) -> None:
        embeddings = None
        cos_sim = None
        avg_sim = None
        try:
            # Set the default CUDA device inside the child process
            torch.cuda.set_device(device_id)
            seeds = self.seeds.to(device_id)
            seeds = F.normalize(seeds.float(), p=2, dim=1)

            embeddings = []
            temp_dict = {}

            for data in load_file_line_by_line(file_path):
                try:
                    embedding = ast.literal_eval(data['vector'])
                    embedding = torch.tensor(embedding).float().to(device_id).requires_grad_(False)

                    if len(embedding) != 768:
                        raise TypeError(f"embedding length is not 768")

                    embeddings.append(embedding)
                    embedding_cpu = embedding.cpu().tolist()
                    temp_dict[data['corpusid']] = {"corpusid": data['corpusid'], "embeddings": embedding_cpu}

                except KeyError as e:
                    self.logger.error(f"No specter2 embedding for corpusid: {data['corpusid']}")

                except TypeError as e:
                    self.logger.error(e)

            if embeddings:
                embeddings = torch.stack(embeddings).to(device_id)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                # Compute cosine similarity
                cos_sim = torch.matmul(embeddings, seeds.T)
                avg_sim = cos_sim.mean(dim=1)
                avg_sim = avg_sim.cpu().tolist()

                # Update temp_dict with similarities
                for i, (corpusid, sub_dict) in enumerate(temp_dict.items()):
                    sub_dict["avg_similarity"] = avg_sim[i]

            output_file = os.path.join(self.output_folder, f"{file_path.stem}_processed.jsonl")
            with open(output_file, 'w') as f_out:
                for sub_dict in temp_dict.values():
                    f_out.write(json.dumps(sub_dict) + '\n')

        finally:
            # Delete tensors if they exist
            if embeddings is not None:
                del embeddings
            if cos_sim is not None:
                del cos_sim
            if avg_sim is not None:
                del avg_sim
            torch.cuda.empty_cache()  # Clear CUDA cache
            gc.collect()  # Force garbage collection to free CPU memory

    def process_files_on_gpu(self, files: list, device_id: int) -> None:
        for file_path in files:
            self.process_file(file_path, device_id)

    def process_all_files(self) -> None:
        # Do not call torch.cuda.device_count() here
        json_files = list(self.input_folder.glob('*.json'))
        num_gpus = int(os.environ.get('NUM_GPUS', 1))  # Use an environment variable or set a default
        gpu_ids = [i for i in range(num_gpus)]
        files_per_gpu = {gpu_id: [] for gpu_id in gpu_ids}

        # Distribute files to GPUs
        for idx, file_path in enumerate(json_files):
            gpu_id = gpu_ids[idx % num_gpus]
            files_per_gpu[gpu_id].append(file_path)

        processes = []
        for gpu_id in gpu_ids:
            p = multiprocessing.Process(target=self.process_files_on_gpu, args=(files_per_gpu[gpu_id], gpu_id))
            p.start()
            processes.append(p)

        # Progress bar setup
        total_files = len(json_files)
        with tqdm(total=total_files, desc='Processing all files') as pbar:
            completed_files = 0
            while any(p.is_alive() for p in processes):
                new_completed_files = total_files - sum(len(files) for files in files_per_gpu.values())
                if new_completed_files > completed_files:
                    pbar.update(new_completed_files - completed_files)
                    completed_files = new_completed_files
                time.sleep(0.5)  # Adjust the sleep time as needed

        for p in processes:
            p.join()