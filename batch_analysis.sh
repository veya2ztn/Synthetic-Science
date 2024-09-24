
sbatch  -p AI4Phys -N1 -c64 --gres=gpu:0  analysis.sh /mnt/petrelfs/zhangtianning.di/dataset/LLM/productive_valid_file_list.uparxive.json.filelist  0 32 128
sbatch  -p AI4Phys -N1 -c64 --gres=gpu:0  analysis.sh /mnt/petrelfs/zhangtianning.di/dataset/LLM/productive_valid_file_list.uparxive.json.filelist 32 32 128
sbatch  -p AI4Phys -N1 -c64 --gres=gpu:0  analysis.sh /mnt/petrelfs/zhangtianning.di/dataset/LLM/productive_valid_file_list.uparxive.json.filelist 64 32 128
sbatch  -p AI4Phys -N1 -c64 --gres=gpu:0  analysis.sh /mnt/petrelfs/zhangtianning.di/dataset/LLM/productive_valid_file_list.uparxive.json.filelist 96 32 128