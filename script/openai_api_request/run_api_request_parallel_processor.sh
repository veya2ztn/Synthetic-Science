#!/bin/bash
#SBATCH -o log/TEST.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/TEST.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

requests_filepath="data/unarXive_quantum_physics/query_full_paper.question.good_questions.jsonl"
save_filepath="data/unarXive_quantum_physics/query_full_paper.question.good_questions.openai.embedding.jsonl"
api_key=
# If need proxy, set the proxy in the command line
# proxy_username=""
# proxy_password=""
proxy="http://zhangtianning.di:Sz3035286@10.1.8.50:33128"

python api_request_parallel_processor.py \
  --requests_filepath $requests_filepath \
  --save_filepath $save_filepath \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 2995 \
  --max_tokens_per_minute 1000000 \
  --token_encoding_name cl100k_base \
  --api_key $api_key \
  --max_attempts 10 \
  --logging_level 20 \
  --proxy $proxy