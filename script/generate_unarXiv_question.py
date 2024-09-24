import torch
import os
import json
import argparse
from tqdm import tqdm

# ROOTDIR='data/unarXive/'
# level     = 2
# root_path = ROOTDIR
# now_path  = [root_path]
# while level>0:
#     new_path = []
#     for root_path in now_path:
#         if os.path.isfile(root_path):continue
#         for sub_name in os.listdir(root_path):
#             sub_path =  os.path.join(root_path,sub_name)
#             new_path.append(sub_path)
#     now_path = new_path
#     level -= 1
# import json

# with open('unarXiv.jsonl_list.json','w') as f:
#     json.dump(now_path,f)

# exit()

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from fastchat.conversation import get_default_conv_template
def read_jsonl(path):
    try:
        with open(path,'r') as f:
            data = [json.load(f)]
    except json.JSONDecodeError:
        with open(path,'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data

def deal_with_papers(papers, save_path, model, tokenizer):
    if os.path.exists(save_path):return 
    json_list = [ ]
    for paper in tqdm(papers):
        abstract = paper['abstract']['text'].strip()
        conv = get_default_conv_template(model_id).copy()
        qs = f"""As a distinguished professor in mathematics and physics, your role involves mentoring Ph.D. students in their journey of understanding complex academic papers. Your task is to carefully read the abstract of a chosen paper and identify ten key concepts that are integral to the study. These concepts should be succinct, expressed as individual words rather than sentences or questions. These key terms will facilitate the students' in-depth exploration and comprehension of the paper's content. Here is the abstract:
        Here is the abstract:"{abstract}"  
        """
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = tokenizer([prompt]).input_ids
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                )
                output_ids = output_ids[0][len(input_ids[0]):]
                outputs = tokenizer.decode(
                    output_ids, skip_special_tokens=True).strip()
        result_json = {'paper_id': paper['paper_id'],
                    'abstract': abstract, 'question': outputs}
        json_list.append(json.dumps(result_json))
    with open(save_path,'w') as f:
        json.dump(json_list,f)
    
model_path = 'checkpoints/vicuna/vicuna-13b-v1.1'
model_id   = 'vicuna'
model_path = os.path.expanduser(model_path)

import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", type=int, required=True)

    args = parser.parse_args()

    with open('FastChat/unarXiv.jsonl_list.json','r') as f:
        all_json_list = json.load(f)

    partition = [int(t) for t in np.linspace(0,len(all_json_list),8)]
    partition_range = np.arange(partition[args.partition_id],partition[args.partition_id+1])
    tokenizer = model = None
    for jsonl_id in partition_range:
        jsonl_path = all_json_list[jsonl_id]
        print(f"deal with partition {jsonl_id} in [{partition_range[0]},{partition_range[-1]}] ==> {jsonl_path}")
        target_path= jsonl_path.replace('/unarXive/','/unarXive_question/')
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):os.makedirs(target_dir)
        papers = read_jsonl(jsonl_path)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()

        deal_with_papers(papers, target_path, model, tokenizer)

