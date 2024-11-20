    
class CreateIndex():
    def __init__(self, input_folder: str, output_dir: str, ):
        

    def create_ind(self, lock):
        for corpusid, values_dict in self.main_dict.items():
            corpusid = int(corpusid)
            
            #key
            key = utils.get_title_abstract_SEPtoken(values_dict)

            #value
            publicationdate = values_dict['publicationdate']
            paperid = values_dict['paper_ids']
            value = f"Corpusid: {corpusid}\nPublicationDate:{publicationdate}\nPaperID: {paperid}"

            #encoded_keys
            embedding = values_dict['embeddings']

            with lock:
                self.keys.append(key)
                self.values.append(value)
                self.encoded_keys.append(embedding)

    def create_ind_all(self) -> None:

        lock = Lock()
        
        json_files = list(self._input_folder.glob('*.json'))
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                    executor.submit(self.create_ind, lock): file_path
                    for file_path in json_files
                }

            for future in tqdm(as_completed(futures), total=len(json_files), desc='Processing all files'):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self._logger.error(f'{file_path} generated an exception: {exc}')