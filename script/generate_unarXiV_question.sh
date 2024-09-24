#export CUDA_VISIBLE_DEVICES=0; nohup python generate_unarXiv_question.py --partition_id 0 > log/runtime.0.log&
export CUDA_VISIBLE_DEVICES=1; nohup python generate_unarXiv_question.py --partition_id 1 > log/runtime.1.log&
export CUDA_VISIBLE_DEVICES=2; nohup python generate_unarXiv_question.py --partition_id 2 > log/runtime.2.log&
export CUDA_VISIBLE_DEVICES=3; nohup python generate_unarXiv_question.py --partition_id 3 > log/runtime.3.log&
#export CUDA_VISIBLE_DEVICES=0; nohup python generate_unarXiv_question.py --partition_id 4 > log/runtime.4.log&
export CUDA_VISIBLE_DEVICES=1; nohup python generate_unarXiv_question.py --partition_id 5 > log/runtime.5.log&
export CUDA_VISIBLE_DEVICES=2; nohup python generate_unarXiv_question.py --partition_id 6 > log/runtime.6.log&
export CUDA_VISIBLE_DEVICES=3; nohup python generate_unarXiv_question.py --partition_id 7 > log/runtime.7.log&