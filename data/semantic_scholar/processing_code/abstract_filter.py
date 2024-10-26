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
            yield json.loads(line)  # Converts each line from JSON string to a Python dictionary
    
class JSONFilter:
    def __init__(self, keywords: List[str], input_directory: str, output_directory: str):
        """
        Initialize the JSON filter.
        
        Args:
            keywords: List of keywords to search for in abstract field
            input_directory: Directory containing JSON files
            output_directory: Directory to save filtered JSON files
        """
        self.keywords = keywords
        self.input_dir = Path(input_directory)
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def contains_keywords(self, abstract: str) -> bool:
        """Check if any keyword exists in the abstract."""
        pattern = r'\b(' + '|'.join(re.escape(keyword) for keyword in self.keywords) + r')\b'
        return bool(re.search(pattern, abstract, re.IGNORECASE))

    def process_file(self, json_file: Path) -> None:
        """Process a single JSON file, keeping only entries with matching keywords in abstract."""
        try:
            filtered_data = []
            counts = 0
            print(f'processing {json_file.name} now')
            for data in load_file_line_by_line(json_file):
                if 'abstract' in data and self.contains_keywords(data['abstract']):
                    filtered_data.append(data)
                    counts += 1

            output_file = self.output_dir / f"filtered_{json_file.name}"
            if len(filtered_data) > 0: 
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2)

            print(f'For {json_file.name}, we have {counts} useful datapoints')
            self.logger.info(f"Processed and filtered: {json_file.name}")
            return
            
        except Exception as e:
            self.logger.error(f"Error processing {json_file.name}: {str(e)}")

    def process_all_files(self, max_workers: int = None) -> None:
        """
        Process all JSON files in parallel using thread pool.
        
        Args:
            max_workers: Maximum number of threads to use
        """
        json_files = list(self.input_dir.glob('*.json'))
        
        if not json_files:
            self.logger.warning(f"No JSON files found in {self.input_dir}")
            return
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_file, json_files)

# Example usage
if __name__ == "__main__":
    # Example configuration
    keywords = ['nlp', 'natural language processing', 'natural-language-processing', 
                'large language model', 'language model', 'llm', 'large language models', 
                'language models', 'llms', 'large-language-model', 'large-language-models', 
                'language-model', 'language-models', 'transformer', 'transformers', 'agent', 
                'agents', 'reasoning', 'natural language', 'instruction tuning', 
                'instruction-tuning', 'singular vision', 'multimodal', 'multi-modal']
    
    input_dir = "2024_10_8/abstracts"
    output_dir = "2024_10_8/abstract_filtered_more"
    
    # Initialize and run the filter
    json_filter = JSONFilter(keywords, input_dir, output_dir)
    json_filter.process_all_files(max_workers = 10)