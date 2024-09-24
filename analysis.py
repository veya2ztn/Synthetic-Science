
import argparse, logging, os
import numpy as np
from tqdm.auto import tqdm
import traceback
from get_ceph_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--index_part", type=int, default=0)
parser.add_argument('--num_parts', type=int, default=1)

parser.add_argument('--mode', type=str, default='analysis')
parser.add_argument('--datapath', type=str, default="http://10.140.52.123:8000/get_data")
parser.add_argument('--model_path', type=str, default='pretrain_weights/vicuna/vicuna-7b-v1.5-16k')
parser.add_argument('--onlinepath', type=str, default="~/dataset/arxiv_uparxiv/")
args = parser.parse_args()
root_path = args.root_path
if os.path.isdir(root_path):
    ###### do not let the program scan the dir ########
    ##### thus the only dir case is that use a dir path like data/archive_json/quant-ph_0004055
    raise NotImplementedError
    all_file_list = [root_path]
elif os.path.isfile(root_path):
    if root_path.endswith('.json'):
        all_file_list = [root_path]
    else:
        with open(root_path,'r') as f:
            all_file_list = [t.strip() for t in f.readlines()]
else:
    all_file_list = [root_path]
    # raise NotImplementedError

index_part= args.index_part
num_parts = args.num_parts 
totally_paper_num = len(all_file_list)
if totally_paper_num > 1:
    divided_nums = np.linspace(0, totally_paper_num - 1, num_parts+1)
    divided_nums = [int(s) for s in divided_nums]
    start_index = divided_nums[index_part]
    end_index   = divided_nums[index_part + 1]
else:
    start_index = 0
    end_index   = 1
    verbose = True

all_file_list = all_file_list[start_index: end_index]

if len(all_file_list)==0:
    print(f"Index {index_part} has no file to process")
    exit()


client = None
### assume arxivid is a path point to the .json files
if "s3:" in args.datapath or "s3:" in args.onlinepath:
    if client is None:
        client = build_client()
analysis = {}
for arxivid in tqdm(all_file_list, leave=False, position=1):

    if arxivid.endswith('.json'): ### then it is a path like 2102/2102.02132/uparxive/2102.02132.json
        arxivname    = arxivid  #<<--- get the arxiv id 
        #2102/2102.02132/metadata.json
        metadataname = os.path.join(os.path.dirname(os.path.dirname(arxivid)), 'metadata.json') 
        #2102/2102.02132/metadata.json
        filepath     = os.path.join(args.datapath  , arxivname)
        metadatapath = os.path.join(args.datapath  , metadataname)

        #2102/2102.02132/uparxive/sentense_question/llama3-8b-instruct.jsonl
        targetpath   = os.path.join(args.onlinepath, os.path.dirname(arxivname), 'sentense_question','llama3-8b-instruct.jsonl')
    else:
        raise NotImplementedError
    
   
   
    # if not check_path_exists(filepath,client):
    #     analysis['no_file'] = analysis.get('no_file',[])+[arxivid]
    #     continue

    if check_path_exists(targetpath,client):
        analysis['already_done'] = analysis.get('already_done',[])+[arxivid]
        continue
    
    analysis['not_finished_yet'] = analysis.get('not_finished_yet',[])+[arxivid]

root_path = "./analysis"
print(root_path)
os.makedirs(root_path,exist_ok=True)
if num_parts > 1:
    for key, val in analysis.items():
        print(f"{key}=>{len(val)}")
        fold = os.path.join(root_path,f"{key.lower()}.filelist.split")
        os.makedirs(fold, exist_ok=True)
        with open(os.path.join(fold,f"{start_index}-{end_index}"), 'w') as f:
            for line in (val):
                f.write(line+'\n')
else:
    #print(analysis)
    for key, val in analysis.items():
        print(f"{key}=>{len(val)}")
        with open(os.path.join(root_path,f"{key.lower()}.filelist"), 'w') as f:
            for line in set(val):
                f.write(line+'\n')
