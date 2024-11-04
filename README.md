# AI-Reviewer

### Statistics
3088499 datapoints from first round of coarse filter

### To run specter2 retrieval:

pip install -r requirements.txt  
cd LitSearch/eval/specter_eval  
python specter_eval.py --topk 20 --query "Bert Model"  
modify the query and topk to return desired number of outputs.