#!/bin/bash
#SBATCH -J analysis
#SBATCH -o log/analysis/%j-analysis.out  
#SBATCH -e log/analysis/%j-analysis.out  
echo `hostname`
# dirpath=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 6) #6个随机字符
# mkdir "/dev/shm/$dirpath"
# ~/s3mount uparxive /dev/shm/$dirpath --profile hansen --max-threads 16 --maximum-throughput-gbps 25 --endpoint-url http://10.140.31.254:80 --prefix json/
declare -a pids

START=$2
SUBCHUNKSIZE=$3
CHUNKSIZE=$4

for ((CPU=0; CPU<SUBCHUNKSIZE; CPU++));
do
    TRUEINDICES=$(($CPU+$START))
    nohup  python scan_and_validation.py --root $1 --index_part $TRUEINDICES  --num_parts $CHUNKSIZE \
    --datapath http://10.140.52.123:8000/get_data --savepath uparxive:s3://uparxive/files \
    > log/convert/thread.$TRUEINDICES.log 2>&1 &
    pids[$CPU]=$!
done 

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All processes have completed."
