import pandas as pd
from mpi4py import MPI
import h5py
from tqdm.auto import tqdm

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

with open('data/unarXive_quantum_physics/paperid_to_title_abstract.jsonl') as f:
    papers = pd.read_json(path_or_buf=f, lines=True)

hdf5_file = 'data/unarXive_full.abstract.h5'

# Use MPI to open the file
with h5py.File(hdf5_file, 'w', driver='mpio', comm=comm) as f:
    for idx, line in tqdm(papers.iterrows(),total=len(papers)):
        # Use rank to determine which jobs this processor should do
        if idx % size == rank:
            paper_id = line.paper_id
            abstract = line.abstract
            title    = line.title
            group_name="abstract"
            if str(group_name) not in f: f.create_group(str(group_name))
            f[f'{group_name}'].create_dataset(str(paper_id), data=abstract)