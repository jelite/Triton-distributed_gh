#!/usr/bin/env bash
set -euo pipefail

# 모델-파라미터를 함께 관리하는 설정 목록
# 포맷: "model_name,in_features,out_features,batch_list(|로 구분)"
configs=(
  "mistral-7b,4096,32768,1|8|16|32|64|128"
  "llama3-8b,4096,128256,1|8|16|32|64|128"
  "llama3-70b,8192,12800,1|8|16|32|64|128"
  "qwen3-30b,2048,151936,1|8|16|32|64|128"
  "gpt-neo,6144,50251,1|8|16|32|64|128"
  "Gemma-7b,3072,256000,1|4|8|16|32|64|128"

)

for conf in "${configs[@]}"; do
  IFS=',' read -r model in_features out_features batches_str <<< "$conf"
  IFS='|' read -r -a batches <<< "$batches_str"

  for batch in "${batches[@]}"; do
    echo "Running: $model, batch=$batch (in=$in_features, out=$out_features)"
    python matmul.py \
      --batch "$batch" \
      --in_features "$in_features" \
      --out_features "$out_features" \
      --model_name "$model"
  done
done