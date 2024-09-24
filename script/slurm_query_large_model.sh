#!/bin/sh
#SBATCH -J query     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-query.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-query.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python -u script/query_large_model_with_large_sentense.py