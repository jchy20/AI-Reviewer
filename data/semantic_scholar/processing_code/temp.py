import os
import json

def count_lines_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f)


input_dir = '2024_10_8/abstract_filtered_more'


counts = 0
for dir in os.listdir(input_dir):
    input_file = os.path.join(input_dir, dir)
    counts += count_lines_in_file(input_file)
    
print(counts)