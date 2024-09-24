from query_pipline.query_full_paper_llama3 import FullPaperQuestionMachine
import torch
from get_ceph_data import *
import re

import json
def get_content_from_json(data):
    title    = None
    abstract = data['abstract']
    content  = []
    for section_num, section_pool in enumerate(data['sections']):
        tag   = section_pool.get('tag', None)
        title = section_pool.get('section_title', None)
        if title is not None:
            content.append(f"# {title}")
        elif tag is not None:
            content.append(f"# Section {tag}")
        else:
            content.append(f"# Section {section_num}")

        section_content = section_pool.get('section_content', [])
        for sec_string in section_content:
            if isinstance(sec_string, list): ### <<---- may a unarxiv json file
                sec_string = "\n".join(sec_string)
            assert isinstance(sec_string, str)
            content.append(sec_string)
        content.append('\n')
    return title, abstract, "\n".join(content)
  

if __name__ == '__main__':
    import argparse, logging, os
    import numpy as np
    from tqdm.auto import tqdm
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--index_part", type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)
    
    parser.add_argument('--mode', type=str, default='analysis')
    parser.add_argument('--model_path', type=str, default='pretrain_weights/Llama3-8B/llama3-8b-instruct')
    parser.add_argument('--datapath', type=str, default="http://10.140.52.123:8000/get_data")
    parser.add_argument('--onlinepath', type=str, default="uparxive:s3://uparxive/files")
    parser.add_argument('--verbose', '-v', action='store_true', help='', default=False)
    parser.add_argument('--redo',  action='store_true', help='', default=False)
    parser.add_argument('--lock',  action='store_true', help='', default=False)
    parser.add_argument('--shuffle',  action='store_true', help='', default=False)
    parser.add_argument('--upload_source_both',  action='store_true', help='', default=False)
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
        raise NotImplementedError
    
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
    if args.shuffle:
        np.random.shuffle(all_file_list)
    all_file_list = all_file_list[start_index: end_index]
    
    if len(all_file_list)==0:
        print(f"Index {index_part} has no file to process")
        exit()
    
    if args.onlinepath is None:
        args.onlinepath = args.datapath

    model_flag = os.path.basename(args.model_path.rstrip('/')).lower()
    client = None
    QuestionMachine = None
    for arxivid in tqdm(all_file_list, leave=False, position=1):
        ### assume arxivid is a path point to the .json files
        if "s3:" in args.datapath or "s3:" in args.onlinepath:
            if client is None:
                client = build_client()

        if arxivid.endswith('.json') or arxivid.endswith('.md'):
            filepath    = arxivid ## assume you
            if arxivid.startswith('archive_json/'):
                arxivid = arxivid[len('archive_json/'):]
            metadataname= os.path.join(os.path.dirname(os.path.dirname(arxivid)), 'metadata.json')
            arxiv_name  = arxivid
        else:
            arxivid  = os.path.split(arxivid)[-1]  #<<--- get the arxiv id
            arxivid  = os.path.split(arxivid)[-1][:-5]
            match = re.search(r"\d{4}", arxivid)
            if match:
                # Print the matched pattern
                date = match.group()
            metadataname = os.path.join(date, arxivid, 'metadata.json')
            arxiv_name   = os.path.join(date, arxivid, 'uparxive', arxivid+'.json')
        
        filepath     = os.path.join(args.datapath  , arxiv_name)
        metadatapath = os.path.join(args.datapath  , metadataname)
        targetpath   = os.path.join(args.onlinepath, os.path.dirname(arxiv_name), 'paper_question', f'{model_flag}.jsonl')

        if args.lock:
            if check_lock_exists(filepath, client):
                tqdm.write(f"[Skip]: has lock ==> {arxiv_name} ")
                continue

        if args.upload_source_both:
            #assert "s3:" in args.onlinepath
            onlinesourcepath   = os.path.join(args.onlinepath, arxiv_name)
            onlinemetadatapath = os.path.join(args.onlinepath, metadataname)
        
        if not check_path_exists(filepath,client):
            tqdm.write(f"[Skip]: no {filepath} ")
            continue

        

        if check_path_exists(targetpath,client) and not args.redo:
            tqdm.write(f"[Skip]: existed {targetpath} ")
            # if args.upload_source_both:
            #     if not check_path_exists(onlinesourcepath,client):
            #         data = read_json_from_path(filepath,client)
            #         if data is not None:
            #             write_json_to_path(data, onlinesourcepath, client)
            #     if not check_path_exists(onlinemetadatapath,client):
            #         metadata     = read_json_from_path(metadatapath, client)
            #         if metadata is not None and not check_path_exists(onlinemetadatapath,client):
            #             write_json_to_path(metadata, onlinemetadatapath, client)
            #tqdm.write(f"[Skip]: has {targetpath} ")
            continue
        data = read_json_from_path(filepath,client)
        
        try:
            data = read_json_from_path(filepath,client)
            if data is None: continue
            metadata     = read_json_from_path(metadatapath, client)
        except:
            traceback.print_exc()
            tqdm.write(f"[fail]==>{filepath}")
            continue
        if check_path_exists(targetpath,client) and not args.redo:
            #tqdm.write(f"[Skip]: has {targetpath} ")
            continue
        
        title, abstract, content = None, None, None
        if filepath.endswith('.json'):
            reterive_content = get_content_from_json(data)
            title, abstract, content = reterive_content
        elif filepath.endswith('.md'):
            content = data['content']
        else:
            raise NotImplementedError('only support json and md file')
        title    = (title or metadata.get('title', None))
        abstract = (abstract or metadata.get('abstract', None))

        if title is None or abstract is None:
            tqdm.write(f"[Skip]: no title or abstract ==> {arxiv_name} ")
            continue
        
        if QuestionMachine is None:
            QuestionMachine = FullPaperQuestionMachine(args.model_path)

        #print(targetpath)
        result = None
        # result = QuestionMachine.ask_question(title, abstract, content)
        try:          
            result = QuestionMachine.ask_question(title, abstract, content)
            
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            tqdm.write(f"[fail]==>{filepath}")
        #tqdm.write(result)
        
        try:
            if result is not None:
                write_json_to_path(result, targetpath, client)

            if args.upload_source_both:
                if data is not None:
                    write_json_to_path(data, onlinesourcepath, client)
                if metadata is not None:
                    write_json_to_path(metadata, onlinemetadatapath, client)
        except:
            traceback.print_exc()
            tqdm.write(f"[fail]==>{filepath}")
  