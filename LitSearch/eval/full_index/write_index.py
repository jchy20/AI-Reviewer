import sys
import os
sys.path.append(os.path.abspath("../../"))
from utils import utils
import json
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

import ast
from threading import Lock
import pickle
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import torch.nn.functional as F
from itertools import islice
import heapq
import re

class WriteIndex:
    def __init__():
        