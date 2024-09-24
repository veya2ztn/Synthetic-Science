for GPU in 0 1 2 3  6 7;
do 
    export CUDA_VISIBLE_DEVICES=$GPU; nohup python -u script/query_large_model.py > log/query_large_model.`hostname`.GPU${GPU}.log&
    sleep 5
done