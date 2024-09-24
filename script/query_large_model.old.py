import torch
import os
import json
import argparse
from tqdm import tqdm
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
)
import pandas as pd
import h5py
import argparse
import numpy as np
#from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
#from ftmodule.model import HFlikemodel
from transformers import StoppingCriteria,StoppingCriteriaList
from .query_llm_model_inference import generate_with_start_kvcache



print("loading csv files...........")
sentense_ids = pd.read_csv("data/unarXive_quantum_physics/unarXive_quantum_physics.clear.sections.id.csv")
print("done~!")
need_question_ids = list(range(len(sentense_ids)))

# print("loading finished ids.........")
# with open("data/unarXive.clear/query.question.results.good_questions.ids.json", 'r') as f:good_question_ids = json.load(f)
# print("done~!")
# good_question_ids=set(good_question_ids)
# need_question_ids=set(list(range(len(sentense_ids)))) - good_question_ids
# need_question_ids = list(need_question_ids)
# print(f"remain {len(need_question_ids)}/{len(sentense_ids)} items")

SAVEPATH = "data/unarXive_quantum_physics/question_results"
sectionsf = h5py.File('data/unarXive_quantum_physics/unarXive_quantum_physics.clear.sections.h5', 'r')
# abstractf = h5py.File('data/unarXive.clear/unarXive.clear.abstract.h5', 'r')
titlef    = h5py.File('data/unarXive_quantum_physics/unarXive_quantum_physics.clear.title.h5', 'r')

print("loading model...........")
#model_path = 'pretrain_weights/fast_t5'
model_path = 'pretrain_weights/vicuna-7b-v1.1'
model, tokenizer = load_model(
    model_path,
    "cuda",
    1,
    load_8bit=False
)

class CharacterStoppingCriteria(StoppingCriteria):
    def __init__(self, none):
        self.stop_token_id_list=[tokenizer.encode("?", add_special_tokens=False)[0],
                                 29973]
    def __call__(self, input_ids, scores, **kwargs):
        assert len(input_ids) == 1
        return input_ids[0][-1] in self.stop_token_id_list
print("done...........")

def deal_with_id(__id):
    #max_length = 128
    tokenizer.padding_side='left'
    _id        = need_question_ids[__id]
    data       = sentense_ids.iloc[_id]
    paper_id   = data['paper_id']
    sentence_id= data['section_num']


    #abstract = abstractf.get(f'abstract/{paper_id}')[()].decode('utf-8')
    title    = titlef.get(f'abstract/{paper_id}')[()].decode('utf-8')
    sentense = sectionsf.get(f'{paper_id}/{sentence_id}')[()].decode('utf-8')
    
    #sentense = " ".join(sentense.split(" ")[:max_length])
    conv = get_conversation_template(model_path)
    #qs = f"""Read below sentence and tell me its type. The answer should be one word and is one of type from ['Author List', 'Reference', 'Content', 'Meaningless']. There is the sentence \"{sentense}\" """
    qs = f"""I have a specific paragraph from a scholarly paper named <{title}>. I need your help to formulate an insightful question based on the information given. The specific paragraph from the paper is:\n\"\"\"\n{sentense} \n\"\"\"\nFollowing the question, provide a one-sentence response that succinctly answers it. The response should be start with "What is". Make the response as short as possible. """
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompts = [conv.get_prompt()]
    #print(prompts[0])
    input_ids = tokenizer(prompts).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    #prompts    = [get_prompt(_id)] #(B=1)
    #input_ids_f= tokenizer(prompts).input_ids
    # tokenizer.pad_token = tokenizer.unk_token
    # inputs_ids = tokenizer.pad({'input_ids': input_ids_f},padding='longest',
    #                             max_length=512,pad_to_multiple_of=8,return_attention_mask=False,
    #                             return_tensors='pt').input_ids.cuda()
    length = len(input_ids[0])
    # with torch.no_grad():
    #     result = model.generate(input_ids, max_length=length+50, stopping_criteria=StoppingCriteriaList([CharacterStoppingCriteria("?")]))
    result = generate_with_start_kvcache(input_ids,tokenizer,{'max_length':128},'cuda',context_len=16000)
    print(result)
    if len(result.shape)==3:result = result[:,0]
    result = tokenizer.decode(result[0][length:], skip_special_tokens=True)
    return {'paper_id':paper_id,'sentence_id':int(sentence_id), 'result':result}


from tqdm import tqdm
total_chunk = 1000
index_range = np.linspace(0, len(need_question_ids), total_chunk+1).astype('int')
import time
if __name__ == '__main__':
    deal_with_id(0)
    # for i in range(total_chunk):
    #     lock_file = f'lock/lock.{i:05d}_{total_chunk:05d}'
    #     if os.path.exists(lock_file):
    #         print(f"{lock_file} exist, continue....")
    #         continue
    #     print(f'create lock file at {lock_file}')
    #     os.system(f'touch {lock_file}')
    #     start = index_range[i]
    #     end   = index_range[i+1]
    #     print(f'deal with sentense from {start} - {end}')
    #     now = time.time()
    #     result = {}
    #     for _id in tqdm(range(start, end)):
    #         try:
    #             result[_id] = deal_with_id(_id)
    #             #print(f"{_id}=>{sentense_ids.iloc[_id]['paper_id']}|{sentense_ids.iloc[_id]['section_num']}==> {result[_id]}")
    #         except:
    #             print(f"{_id}=>{sentense_ids.iloc[_id]['paper_id']}|{sentense_ids.iloc[_id]['section_num']}==> fail!!! ")
    #     print(f"cost {time.time() - now}")
    #     with open(f"{SAVEPATH}/type_{start:08d}_{end:08d}.json",'w') as ff:
    #         json.dump(result,ff)

        

