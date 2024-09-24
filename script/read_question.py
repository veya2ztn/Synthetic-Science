from tqdm import tqdm
import multiprocessing
import json
import re,os
import csv
import concurrent.futures
# assuming your function find_subsentence is defined somewhere
def find_subsentence(sentence):
    match = re.search(r'(What|How|Why|Where|Can|Could|Would|Is|Will).*\?', sentence)
    if match:
        return match.group(0)
    else:
        return None


def deal_with_json_file(path):
    try:
        with open(path, 'r') as f:data = json.load(f)
    except:
        print(f"fail at {path}")
        return [],[]
    good_questions = []
    bad_questions = []

    for question_id, metadata in data.items():
        try:
            question = metadata['result']
        except:
            print(metadata)
            continue
        if not question:continue
        true_question = find_subsentence(question.strip())
        if true_question is None:
            bad_questions.append(
                [question_id, metadata['paper_id'], metadata['sentence_id'], metadata['result']])
        else:
            good_questions.append(
                [question_id, metadata['paper_id'], metadata['sentence_id'], true_question])

    return good_questions, bad_questions


def multiprocessing_handler(path_list, max_workers):
    good_questions = []
    bad_questions = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            deal_with_json_file, path): path for path in path_list}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            good_questions.extend(result[0])
            bad_questions.extend(result[1])

    return good_questions, bad_questions

ROOTDIR = '/mnt/petrelfs/weigengchen/llm_doc/llm_train/uniem-train/ztn_dataset/LLM/unarXiv.all_text/cs.CV'
SAVEDIR = '/mnt/petrelfs/weigengchen/llm_doc/llm_train/uniem-train/data/unarXiv.all_text/cs.CV/question_per_paper'
SOURCEDIR = os.path.join(ROOTDIR, "full_paper_question_results")
path_list = [os.path.join(SOURCEDIR,p) for p in os.listdir(SOURCEDIR)][:4]

good_questions, bad_questions = multiprocessing_handler(path_list,64)
print(len(good_questions))
print(len(bad_questions))

import pandas as pd
good_questions = pd.DataFrame(good_questions,columns=['question_id','paper_id','sentense_id','question'])
good_questions.to_csv(os.path.join(SAVEDIR, 'query.question.results.good_questions.csv'), 
                      quoting=csv.QUOTE_NONE,
                      escapechar='\\')



import json
with open(os.path.join(SAVEDIR, 'query.question.results.good_questions2.ids.json'),'w') as f:
    json.dump(good_questions['question_id'].tolist(),f)


bad_questions= pd.DataFrame(bad_questions,columns=['question_id','paper_id','sentense_id','question'])
bad_questions.to_csv(os.path.join(
    SAVEDIR, 'query.question.results.bad_questions2.csv'),
    quoting=csv.QUOTE_NONE,
    escapechar='\\')
