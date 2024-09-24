import pandas as pd
import os
from tqdm.auto import tqdm
import json
ROOTDIR = 'data/unarXive_quantum_physics/'
df= pd.read_csv(os.path.join(ROOTDIR,"query_full_paper.question.good_questions.csv"))
# convert the DataFrame to JSONL format
jsonl_data = []
for index, row in tqdm(df.iterrows(),total= len(df)):
    jsonl_data.append(json.dumps({
        "model": "text-embedding-ada-002",
        "input": row['question'],
        "metadata": {
            "question_id": row['question_id'],
            "question_row":index,
            "paper_id": row['paper_id'],
            "question_type": row['question_type'],
        }
    }))
jsonl_string = '\n'.join(jsonl_data)

with open('data/unarXive_quantum_physics/query_full_paper.question.good_questions.jsonl', 'w') as f:
    f.write(jsonl_string)