import time
import os  # Added to handle file paths
import requests
import shutil
import gzip
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import threading

class S2Download():
    def __init__(self, rate_limit: int, file_urls: list, category: str, output_folder: str, headers: dict):
        #api rate limit
        self.rate_limit = rate_limit
        self.min_request_interval = 1.0 / self.rate_limit
        self.LRT = 0

        #files
        self.file_urls = file_urls
        self.category = category
        self.output_folder = output_folder
        self.headers = headers
        
        #others
        self.index_counter = itertools.count()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_file(self, file_url, lock):
        # Use the next value of the atomic counter as the index for this file
        idx = next(self.index_counter)
        file_name = f'{self.category}_file{idx}.json'
        
        # Create the full file path in the specified folder
        file_path = os.path.join(self.output_folder, file_name)
        
        # logger.info(f"Downloading file {idx + 1}: {file_url}")
        success = False
        retries = 0
        max_retries = 5  # Maximum number of retries for each file
        delay = 1  # Initial delay for exponential backoff

        while not success and retries < max_retries:
            with lock:
                current_time = time.time()
                elapsed = current_time - self.LRT
                if elapsed < self.min_request_interval:
                    sleep_time = self.min_request_interval - elapsed
                    time.sleep(sleep_time)
                
                self.LRT = time.time()  # Update last request time

            try:
                with requests.get(file_url, headers=self.headers, stream=True) as r:
                    r.raise_for_status()  # Raise an exception if the download failed
                    with gzip.open(r.raw, 'rb') as decompressed_stream:
                        with open(file_path, 'wb') as out_file:  # Save to file_path
                            shutil.copyfileobj(decompressed_stream, out_file)
                success = True  # Download succeeded
                # logger.info(f"Successfully downloaded {file_path}.")
            except Exception as e:
                self.logger.error(f"Error downloading {file_url}: {e}")
                retries += 1
                if retries < max_retries:
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff: double the delay
                else:
                    self.logger.into("Max retries reached. Skipping this file.")
        return success

    # Download files with multi-threading
    def download_all_files_concurrently(self):
        max_workers = 10  # Adjust based on your system's capabilities=
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.download_file, file_url, lock): file_url 
                for file_url in self.file_urls
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(self.file_urls), desc=f'Downloading {self.category}'):
                file_url = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        self.logger.info(f"Failed to download: {file_url}")
                except Exception as e:
                    self.logger.error(f"Error with future {file_url}: {e}")


endpoint = 'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/embeddings-specter_v2'
headers = {
    'x-api-key': 'PB2IUQCkgU7VqjOeJ1v7W3h688WDRxrnaialTY1q',
    'Accept': 'application/json'
}
category = 'embeddings_v2'
response = requests.get(endpoint, headers=headers).json()
embeddings_v2_files = response.get('files', [])

output_folder = os.path.join('2024-11-12', category)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

embeddings_v2_download = S2Download(1, embeddings_v2_files, category, output_folder, headers)
embeddings_v2_download.download_all_files_concurrently()