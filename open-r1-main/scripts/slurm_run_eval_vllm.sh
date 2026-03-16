#!/bin/bash
#SBATCH --job-name=lighteval_math
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
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
MODEL_DIR=/home1/doyoonkim/projects/RAC/open-r1-main/models/DeepSeek-R1-Distill-Qwen-1_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl
MODEL_TAG=$(basename "$MODEL_DIR")

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=4,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=./eval_results_cot/${MODEL_TAG}

########################################
# 4. 실행
########################################
lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
  --output-dir "$OUTPUT_DIR" \
  --save-details

echo "Evaluation finished."