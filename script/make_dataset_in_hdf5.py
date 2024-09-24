import pandas as pd
from tqdm.auto import tqdm
import h5py
with open('data/unarXive_quantum_physics/paperid_to_title_abstract.jsonl') as f:
    papers = pd.read_json(path_or_buf=f, lines=True)
hdf5_file = 'data/unarXive_full.clear.abstract.h5'
with h5py.File(hdf5_file, 'w') as f:

    

    for idx, line in tqdm(papers.iterrows(),total=len(papers)):
        paper_id = line.paper_id
        abstract = line.abstract
        title    = line.title
        group_name="abstract"
        if str(group_name) not in f:f.create_group(str(group_name))
        f[f'{group_name}'].create_dataset(str(paper_id), data=abstract)