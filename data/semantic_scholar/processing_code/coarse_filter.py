import os
import json
import re
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_file_line_by_line(file_path):
    '''
    return an iterable file
    '''
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)  # Converts each line from JSON string to a Python dictionary

def check_abstract(abstract, keywords):
    """
    Inputs:
    abstract: text of abstract
    keywords: a list of keywords
    return whether a abstract contains at least one keyword
    """

    pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
    return bool(re.search(pattern, abstract, re.IGNORECASE))

def combine_dict(abstract_dict, dicts, category_names):
    """
    Inputs:
    abstract: text of abstract
    dicts: a dict of dicts, keys are category names and values are the retreived corresponding parts of the paper
    Return one combined dict
    """
    temp_dict = abstract_dict.copy()
    for category in category_names:
        if category in dicts and dicts[category]:
            temp_dict.update(dicts[category])

    return temp_dict

def fetch_from_file_with_stop(file_path, corpusid, result_queue, stop_event):
    """
    Try to find matching record from one single json file
    """
    print(f'processcing file {file_path}')
    for item in load_file_line_by_line(file_path):
        if stop_event.is_set():
            return  # Another thread has already found the result, stop searching

        if 'corpusid' in item and item['corpusid'] == corpusid:
            result_queue.put(item)  # Put the result in the queue
            stop_event.set()  # Signal other threads to stop searching
            return


def write_results(output_queue, output_file):
    """
    Writes the combined records from the output queue to the final output file.
    """
    with open(output_file, 'w') as f:
        while True:
            record = output_queue.get()
            if record is None:
                break
            f.write(json.dumps(record) + '\n')

def process_category(category_name, category_path, corpusid, num_workers):
    """
    Process a single category using num_workers threads.
    Each worker processes a different file in the category.
    Use a ThreadPoolExecutor to manage worker threads.
    """
    result_queue = Queue()
    stop_event = threading.Event()

    # Get the list of files in the category
    files = [os.path.join(category_path, file) for file in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, file))]

    # Use a ThreadPoolExecutor to limit the number of concurrent threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file_path in files:
            if stop_event.is_set():
                break  

            futures.append(executor.submit(fetch_from_file_with_stop, file_path, corpusid, result_queue, stop_event))

        # Wait for the results
        for future in as_completed(futures):
            if stop_event.is_set():
                break  

    if not result_queue.empty():
        return result_queue.get()  
    else:
        return None 


def process_file(file_path, output_queue, categories, keywords, category_names, num_workers):
    """
    Processes each file in the abstracts folder, checks the abstract for keywords, 
    fetches corresponding data from categories one by one, and combines the dictionaries.
    """
    for abstract_record in load_file_line_by_line(file_path):
        corpusid = abstract_record.get('corpusid')

        # Check if the abstract contains any of the keywords
        if 'abstract' in abstract_record and check_abstract(abstract_record['abstract'], keywords):
            print('it did run')
            # Dictionary to store the fetched components from different categories
            fetched_data = {}

            # Process each category one by one
            for category in category_names:
                if category == 'abstracts':
                    continue
                category_path = categories[category]
                result = process_category(category, category_path, corpusid, num_workers)

                if result:
                    fetched_data[category] = result

            # Combine the abstract dict with data from other categories
            combined_record = combine_dict(abstract_record, fetched_data, category_names)
            # Push the combined record to the output queue for writing
            output_queue.put(combined_record)



base_folder = '2024_10_8'
abstracts_folder = os.path.join(base_folder, 'abstracts')

keywords = ['artificial intelligence', 'machine learning', 'deep learning']
category_names = ['abstracts', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 's2orc', 'tldrs']

# category_names = ['abstracts', 'authors', 'citations', 'embeddings_v1', 'embeddings_v2', 'paper_ids', 'papers', 'publication_venues', 's2orc', 'tldrs']

categories = {}

for category in category_names: 
    if category == 'abstracts':
        continue
    categories[category] = os.path.join(base_folder, category)


print(categories)


num_workers = 10

total = 0

for idx, dir in enumerate(os.listdir(abstracts_folder)):

    output_queue = Queue()

    file_path = os.path.join(abstracts_folder, dir)
    print(f'processing {file_path}')
    process_file(file_path, output_queue, categories, keywords, category_names, num_workers)

    output_name = f'combined_file_{idx}.json'
    folder = os.path.join(base_folder, 'combined')
    output_file_path = os.path.join(folder, output_name)
    
    with open(output_file_path, 'w') as output_file:
        while not output_queue.empty():
            combined_record = output_queue.get()
            output_file.write(json.dumps(combined_record) + '\n')
            total += 1
    print(f'for {file_path}, we have {total} accepted papers')

print(f'total paper: {total}')