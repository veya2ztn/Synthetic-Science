import time
import numpy as np
import h5py
import pandas as pd
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
)
from tqdm import tqdm
import argparse
import json
import os
import torch
from vllm import LLM,SamplingParams

ROOTDIR = 'data/unarXiv.all_text/cs.CV'
SAVEPATH = os.path.join(ROOTDIR, "question_per_sentense/source")


sentense_ids_path = os.path.join(ROOTDIR,"cs.CV.sentense.ids.csv")
print(f"loading csv files from {sentense_ids_path} ........")
sentense_ids = pd.read_csv(sentense_ids_path)
print("done~!")
need_question_ids = list(range(len(sentense_ids)))

# print("loading finished ids.........")
# with open(os.path.join(ROOTDIR, "query.question.results.good_questions.ids.json"), 'r') as f:
#     good_question_ids = json.load(f)

# print("done~!")
# good_question_ids=set(good_question_ids)
# need_question_ids=set(list(range(len(sentense_ids)))) - good_question_ids
# need_question_ids = list(need_question_ids)
# print(f"remain {len(need_question_ids)}/{len(sentense_ids)} items")



sectionsf = h5py.File(os.path.join(ROOTDIR, "cs.CV.clear.sections.h5"), 'r')
# abstractf = h5py.File('data/unarXive.clear/unarXive.clear.abstract.h5', 'r')
titlef = h5py.File(os.path.join(ROOTDIR, "cs.CV.clear.title.h5"), 'r')

print("loading model...........")
# model_path = 'pretrain_weights/fast_t5'
model_path = 'pretrain_weights/vicuna-7b-v1.1'
# model, tokenizer = load_model(
#     model_path,
#     "cuda",
#     1,
#     load_8bit=False
# )

llm       = LLM(model=model_path)

def deal_with_id(__id):
    _id = need_question_ids[__id]
    data = sentense_ids.iloc[_id]

    paper_id = data['paper_id']
    sentence_id = data['section_num'] 
    
    #abstract = abstractf.get(f'abstract/{paper_id}')[()].decode('utf-8')
    title = titlef.get(f'title/{paper_id}')[()].decode('utf-8').replace('\n'," ")

    sentense = sectionsf.get(f'{paper_id}/{sentence_id}')[()].decode('utf-8').replace('\n', " ")

    #sentense = " ".join(sentense.split(" ")[:max_length])
    conv = get_conversation_template(model_path)
    #qs = f"""Read below sentence and tell me its type. The answer should be one word and is one of type from ['Author List', 'Reference', 'Content', 'Meaningless']. There is the sentence \"{sentense}\" """
    qs = f"""I have a specific paragraph from a scholarly paper named <{title}>. I need your help to formulate an insightful question based on the information given. The specific paragraph from the paper is:\n\"\"\"\n{sentense} \n\"\"\"\nFollowing the question, provide a one-sentence response that succinctly answers it. The response should be start with "What is". Make the response as short as possible. The response should not be general like "What is the main purpose" """
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    
    sampling_params = SamplingParams(stop=["?"],max_tokens=150)
    result = llm.generate(conv.get_prompt(), sampling_params, use_tqdm=False)
    result = result[0].outputs[0]
    
    output_string = result.text
    if str(result.finish_reason) == "stop":
        output_string = output_string + "?"

    return {'paper_id': paper_id, 'sentence_id': int(sentence_id), 'result': output_string}


total_chunk = 2000
index_range = np.linspace(0, len(need_question_ids),
                          total_chunk+1).astype('int')
cost_list = []
if __name__ == '__main__':
    for i in tqdm(range(total_chunk)):
        lock_file = f'lock/lock.{i:05d}_{total_chunk:05d}'
        if os.path.exists(lock_file):
            print(f"{lock_file} exist, continue....")
            continue
        print(f'create lock file at {lock_file}')
        os.system(f'touch {lock_file}')
        start = index_range[i]
        end = index_range[i+1]
        print(f'deal with sentense from {start} - {end}')
        now = time.time()
        result = {}
        for _id in tqdm(range(start, end)):
            try:
                result[_id] = deal_with_id(_id)
                #print(f"{_id}=>{sentense_ids.iloc[_id]['paper_id']}|{sentense_ids.iloc[_id]['section_num']}==> {result[_id]}")
            except:
                print(f"{_id}=>{sentense_ids.iloc[_id]['paper_id']}|{sentense_ids.iloc[_id]['section_num']}==> fail!!! ")
                torch.cuda.empty_cache()
        if not os.path.exists(SAVEPATH):
            os.makedirs(SAVEPATH)
        print(f"cost {time.time() - now}")
        with open(f"{SAVEPATH}/type_{start:08d}_{end:08d}.json", 'w') as ff:
            json.dump(result, ff)
