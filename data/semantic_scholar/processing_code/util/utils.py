import json
from pathlib import Path
import concurrent.futures
import logging
from typing import List
import re

def load_file_line_by_line(file_path):
    '''
    return an iterable file
    '''
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)  