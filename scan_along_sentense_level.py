#from query_pipline.query_simple import SimpleSentenseQvLLM
from query_pipline.query_simple_llama3 import SimpleSentenseQvLLM
import re,torch
from get_ceph_data import *
import re

def get_content_from_json(data):
    
    title    = None
    abstract = data.get('abstract',None)
    sentenses= []
    sentense_id = 0
    #### we will flatten the sentense in a list, but record the location postion and eventually revert the list of question in a same structure
    for section_num, section_pool in enumerate(data['sections']):
        ### for each pool, it is nice if we can concat some short sentense to the ahead sentense and make the length of sentense more unique
        pool_sentense = []
        section_content = section_pool.get('section_content', [])
        for sec_string in section_content:
            if isinstance(sec_string, list): ### <<---- may a unarxiv json file
                sec_string = "\n".join(sec_string)
            assert isinstance(sec_string, str)
            pool_sentense.append([str(sentense_id), sec_string])
            sentense_id+=1
        ## now lets concat the sentense
        ### please finish the merge code
        
        merged_sentenses = []
        for now_sentense_id, sentense in pool_sentense:
            word_count = len(sentense.split())
            
            if word_count < 50 and len(merged_sentenses) > 0:
                prev_sentense_id, prev_sentense = merged_sentenses[-1]
                prev_sentense_id = f"{prev_sentense_id}|{now_sentense_id}"
                prev_sentense = prev_sentense +'\n' + sentense
                merged_sentenses[-1] = (prev_sentense_id, prev_sentense)
            else:
                merged_sentenses.append((now_sentense_id, sentense))
        sentenses.extend(pool_sentense)
    return title, sentenses


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
    parser.add_argument('--model_path', type=str)  # 'pretrain_weights/vicuna/vicuna-7b-v1.5-16k' 
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

        if not arxivid.endswith('.json'):
            arxivid     = os.path.split(arxivid)[-1]  #<<--- get the arxiv id
            arxivid  = os.path.split(arxivid)[-1][:-5]
            match = re.search(r"\d{4}", arxivid)
            if match:
                # Print the matched pattern
                date = match.group()
            metadataname = os.path.join(date, arxivid, 'metadata.json')
            arxiv_name   = os.path.join(date, arxivid, 'uparxive', arxivid+'.json')
        else:
            filepath    = arxivid ## assume you
            if arxivid.startswith('archive_json/'):
                arxivid = arxivid[len('archive_json/'):]
            metadataname= os.path.join(os.path.dirname(os.path.dirname(arxivid)), 'metadata.json')
            arxiv_name  = arxivid
            
            
        
        filepath     = os.path.join(args.datapath  , arxiv_name)
        metadatapath = os.path.join(args.datapath  , metadataname)
        targetpath   = os.path.join(args.onlinepath, os.path.dirname(arxiv_name), 'sentense_question', f'{model_flag}.jsonl')

        
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
        
        try:
            data = read_json_from_path(filepath,client)
            if data is None: continue
            metadata     = read_json_from_path(metadatapath, client)
        except:
            traceback.print_exc()
            tqdm.write(f"[fail]==>{filepath}")
            continue

        reterive_content = get_content_from_json(data)
        title, sentenses = reterive_content
        if title is None:
            title        = metadata.get('title', None) if metadata else None
            abstract     = metadata.get('abstract', None) if metadata else None
        if title is None:
            tqdm.write(f"Error: {filepath} has no title. Retry after add the metadata")
            continue
        alltitles    = [title]*len(sentenses)
        allabstracts = [abstract]*len(sentenses)
        sentense_ids = [sentense_id for sentense_id, _ in sentenses]
        sentenses    = [sentense for _, sentense in sentenses]

        
        if QuestionMachine is None:
            QuestionMachine = SimpleSentenseQvLLM(args.model_path)

        batchsize = 16
        result = None
        while batchsize > 0 and result is None:
            try:    
                result = QuestionMachine.ask_question_batch(alltitles, allabstracts, sentenses, sentense_ids, batch_size = batchsize)
                break
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    batchsize = batchsize//2
                    tqdm.write(f"""
                                half the batchsize to {batchsize}, try again!!
                                """)
                    torch.cuda.empty_cache()
                    continue
                else:
                    traceback.print_exc()
                    tqdm.write(f"[fail]==>{filepath}")
                    break
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
            
