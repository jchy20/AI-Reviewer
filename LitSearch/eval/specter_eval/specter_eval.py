import sys
import os
sys.path.append(os.path.abspath("../../"))
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from eval.retrieval.kv_store import KVStore
from eval.retrieval.specter2 import SPECTER2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

######################## initialize retriever model
index_path = '/usr/project/xtmp/ai_reviewer/index/list_specter2.specter2'
specter2_tensor = SPECTER2(None, None, None)
specter2_tensor = specter2_tensor.load(index_path)
specter2_tensor.save_as_tensor = True
specter2_tensor.encoded_keys = torch.stack(specter2_tensor.encoded_keys).to("cuda")
#########################

model_path = "allenai/specter2_base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAdapterModel.from_pretrained(model_path)
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.to("cuda")

query1 = "Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts[SEP]By providing external information to large language models (LLMs), tool augmentation (including retrieval augmentation) has emerged as a promising solution for addressing the limitations of LLMs' static parametric memory. However, how receptive are LLMs to such external evidence, especially when the evidence conflicts with their parametric memory? We present the first comprehensive and controlled investigation into the behavior of LLMs when encountering knowledge conflicts. We propose a systematic framework to elicit high-quality parametric memory from LLMs and construct the corresponding counter-memory, which enables us to conduct a series of controlled experiments. Our investigation reveals seemingly contradicting behaviors of LLMs. On the one hand, different from prior wisdom, we find that LLMs can be highly receptive to external evidence even when that conflicts with their parametric memory, given that the external evidence is coherent and convincing. On the other hand, LLMs also demonstrate a strong confirmation bias when the external evidence contains some information that is consistent with their parametric memory, despite being presented with conflicting evidence at the same time. These results pose important implications that are worth careful consideration for the further development and deployment of tool- and retrieval-augmented LLMs. Resources are available at this https URL."
query2 = "ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence[SEP]Retrieval augmented generation (RAG) is frequently used to mitigate hallucinations and provide up-to-date knowledge for large language models (LLMs). However, given that document retrieval is an imprecise task and sometimes results in erroneous or even harmful content being presented in context, this raises the question of how LLMs handle retrieved information: If the provided content is incorrect, does the model know to ignore it, or does it recapitulate the error? Conversely, when the model's initial response is incorrect, does it always know to use the retrieved information to correct itself, or does it insist on its wrong prior response? To answer this, we curate a dataset of over 1200 questions across six domains (e.g., drug dosages, Olympic records, locations) along with content relevant to answering each question. We further apply precise perturbations to the answers in the content that range from subtle to blatant errors. We benchmark six top-performing LLMs, including GPT-4o, on this dataset and find that LLMs are susceptible to adopting incorrect retrieved content, overriding their own correct prior knowledge over 60% of the time. However, the more unrealistic the retrieved content is (i.e. more deviated from truth), the less likely the model is to adopt it. Also, the less confident a model is in its initial response (via measuring token probabilities), the more likely it is to adopt the information in the retrieved content. We exploit this finding and demonstrate simple methods for improving model accuracy where there is conflicting retrieved content. Our results highlight a difficult task and benchmark for LLMs -- namely, their ability to correctly discern when it is wrong in light of correct retrieved content and to reject cases when the provided content is incorrect."
query3 = "Truth-Aware Context Selection: Mitigating Hallucinations of Large Language Models Being Misled by Untruthful Contexts[SEP]Although Large Language Models (LLMs) have demonstrated impressive text generation capabilities, they are easily misled by untruthful contexts provided by users or knowledge augmentation tools, leading to hallucinations. To alleviate LLMs from being misled by untruthful context and take advantage of knowledge augmentation, we propose Truth-Aware Context Selection (TACS), a lightweight method to adaptively recognize and mask untruthful context from the inputs. TACS begins by performing truth detection on the input context, leveraging the parameterized knowledge within the LLM. Subsequently, it constructs a corresponding attention mask based on the truthfulness of each position, selecting the truthful context and discarding the untruthful context. Additionally, we introduce a new evaluation metric, Disturbance Adaption Rate, to further study the LLMs' ability to accept truthful information and resist untruthful information. Experimental results indicate that TACS can effectively filter untruthful context and significantly improve the overall quality of LLMs' responses when presented with misleading information."

situated_faithfull = "Enhancing Large Language Models' Situated Faithfulness to External Contexts[SEP]Large Language Models (LLMs) are often augmented with external information as contexts, but this external information can sometimes be inaccurate or even intentionally misleading. We argue that robust LLMs should demonstrate situated faithfulness, dynamically calibrating their trust in external information based on their confidence in the internal knowledge and the external context. To benchmark this capability, we evaluate LLMs across several QA datasets, including a newly created dataset called RedditQA featuring in-the-wild incorrect contexts sourced from Reddit posts. We show that when provided with both correct and incorrect contexts, both open-source and proprietary models tend to overly rely on external information, regardless of its factual accuracy. To enhance situated faithfulness, we propose two approaches: Self-Guided Confidence Reasoning (SCR) and Rule-Based Confidence Reasoning (RCR). SCR enables models to self-access the confidence of external information relative to their own internal knowledge to produce the most accurate answer. RCR, in contrast, extracts explicit confidence signals from the LLM and determines the final answer using predefined rules. Our results show that for LLMs with strong reasoning capabilities, such as GPT-4o and GPT-4o mini, SCR outperforms RCR, achieving improvements of up to 24.2% over a direct input augmentation baseline. Conversely, for a smaller model like Llama-3-8B, RCR outperforms SCR. Fine-tuning SCR with our proposed Confidence Reasoning Direct Preference Optimization (CR-DPO) method improves performance on both seen and unseen datasets, yielding an average improvement of 8.9% on Llama-3-8B. In addition to quantitative results, we offer insights into the relative strengths of SCR and RCR. Our findings highlight promising avenues for improving situated faithfulness in LLMs. The data and code are released."

candidates = [query1, query2, query3, situated_faithfull]
inputs = tokenizer(candidates, padding=True, truncation=True,
                    return_tensors="pt", 
                    return_token_type_ids=False,
                    max_length=512).to("cuda")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :].detach().clone().requires_grad_(False)

candidate_embed = embeddings[0:3]
sf_embed = embeddings[-1]

scores20, kv20 = specter2_tensor.query(situated_faithfull, 500, return_sim_score=True)

scores3 = F.cosine_similarity(candidate_embed, sf_embed)

scores20 = torch.sort(scores20, descending=True).values
scores3 = torch.sort(scores3, descending=True).values


i, j = 0, 0
while i < len(scores20) and scores20[i] > scores3[0]:
    i += 1

print(i)
print(scores20[i])


