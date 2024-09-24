#!/bin/bash
#SBATCH -J QuerySentence
#SBATCH -o log/%j-QuerySentence.out  
#SBATCH -e log/%j-QuerySentence.out  
# dirpath=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 6) #6个随机字符
# mkdir "/dev/shm/$dirpath"
# ~/s3mount uparxive /dev/shm/$dirpath --profile hansen --max-threads 16 --maximum-throughput-gbps 25 --endpoint-url http://10.140.31.254:80 --prefix json/
# python scan_along_sentense_level.py --root_path $1 --index_part $2 --num_parts $3 --upload_source_both --datapath http://10.140.52.123:8000/get_data --onlinepath /dev/shm/$dirpath
# fusermount -u "/dev/shm/$dirpath"
echo `hostname`
# python scan_along_sentense_level.py --root_path $1 --index_part $2 --num_parts $3 --upload_source_both --datapath http://10.140.52.123:8000/get_data \
# --onlinepath uparxive:s3://uparxive/files \
# --model_path pretrain_weights/Llama3-8B/llama3-8b-instruct 
python scan_along_paper_level.py --root_path $1 --index_part $2 --num_parts $3 --upload_source_both --datapath http://10.140.52.123:8000/get_data \
--onlinepath uparxive:s3://uparxive/files \
--model_path pretrain_weights/Llama3-8B/llama3-8b-instruct 
#python scan_along_paper_level.py --root_path $1 --index_part $2 --num_parts $3 --datapath uparxive:s3://uparxive/json --onlinepath uparxive:s3://uparxive/json --shuffle
