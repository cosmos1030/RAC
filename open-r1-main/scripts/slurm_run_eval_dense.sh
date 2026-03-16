#!/bin/bash
#SBATCH --job-name=lighteval_math
#SBATCH --partition=4A100
#SBATCH --qos=4A100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j_eval.out
#SBATCH -t 1-00:00:00

set -euo pipefail
mkdir -p logs

########################################
# 1. Conda
########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

########################################
# 2. CUDA 안정화 옵션
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

########################################
# 3. 모델 설정
########################################
MODEL_DIR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_TAG=$(basename "$MODEL_DIR")

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=4,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=./eval_results_dense/${MODEL_TAG}

########################################
# 4. 실행
########################################
lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
  --output-dir "$OUTPUT_DIR" \
  --save-details

echo "Evaluation finished."