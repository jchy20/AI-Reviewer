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

# Constants

headers = {
    'x-api-key': '',
    'Accept': 'application/json'
}
# List of dataset endpoints
endpoints = [
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/abstracts',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/authors',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/citations',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/embeddings-specter_v1',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/embeddings-specter_v2',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/paper-ids',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/papers',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/publication-venues',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/s2orc',
    'https://api.semanticscholar.org/datasets/v1/release/2024-11-12/dataset/tldrs'
]
# Dictionary to store the results for each dataset
categories = {}
category_names = ['abstracts', 'authors', 'citations', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 'publication_venues', 's2orc', 'tldrs']
# Loop through each endpoint, making requests with a 2-second delay
for idx, endpoint in enumerate(endpoints):
    try:
        response = requests.get(endpoint, headers=headers).json()
        categories[category_names[idx]] = response.get('files', [])
        print(f"Successfully fetched data from {endpoint}")
    except Exception as e:
        print(f"Error fetching data from {endpoint}: {e}")
    
    # Wait for 2 seconds before the next request, except for the last iteration
    if idx < len(endpoints) - 1:
        time.sleep(2)


for category_name, url in categories.items():
    print(f'{category_name}: {len(url)}')

RATE_LIMIT = 1  # Max requests per second
MIN_REQUEST_INTERVAL = 1.0 / RATE_LIMIT  # Minimum interval between requests in seconds
LRT = 0


# File download function
def download_file(file_url, dataset, headers, folder_path, index_counter, logger, lock):
    # Use the next value of the atomic counter as the index for this file
    idx = next(index_counter)
    file_name = f'{dataset}_file{idx}.json'
    
    # Create the full file path in the specified folder
    file_path = os.path.join(folder_path, file_name)
    
    # logger.info(f"Downloading file {idx + 1}: {file_url}")
    success = False
    retries = 0
    max_retries = 5  # Maximum number of retries for each file
    delay = 1  # Initial delay for exponential backoff

    while not success and retries < max_retries:
        current_time = time.time()
        elapsed = current_time - LRT
        if elapsed < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - elapsed
            logger.error(f"Rate limit enforced. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        
        with lock:
            global LRT
            LRT = time.time()  # Update last request time
        try:
            with requests.get(file_url, headers=headers, stream=True) as r:
                r.raise_for_status()  # Raise an exception if the download failed
                with gzip.open(r.raw, 'rb') as decompressed_stream:
                    with open(file_path, 'wb') as out_file:  # Save to file_path
                        shutil.copyfileobj(decompressed_stream, out_file)
            success = True  # Download succeeded
            # logger.info(f"Successfully downloaded {file_path}.")
        except Exception as e:
            logger.error(f"Error downloading {file_url}: {e}")
            retries += 1
            if retries < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff: double the delay
            else:
                logger.into("Max retries reached. Skipping this file.")
    return success


# Download files with multi-threading
def download_all_files_concurrently(files, dataset, headers, folder_path, index_counter):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    max_workers = 10  # Adjust based on your system's capabilities=
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_file, file_url, dataset, headers, folder_path, index_counter, logger, lock): file_url for file_url in files}
        
        for future in tqdm(as_completed(future_to_file), total=len(files), desc=f'Downloading {category}'):
            file_url = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    logger.info(f"Failed to download: {file_url}")
            except Exception as e:
                logger.error(f"Error with future {file_url}: {e}")


folder_path1 = '2024-11-12'

names = ['embeddings_v2']
for category, url in categories.items():
    if category in names:
        folder_path = os.path.join(folder_path1, category)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Thread-safe counter for file indexing
        index_counter = itertools.count()
        download_all_files_concurrently(url, category, headers, folder_path, index_counter)