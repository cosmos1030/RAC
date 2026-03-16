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

# 로그 폴더 생성
mkdir -p logs

# 1. 환경 설정 및 Conda 활성화
# (주의: conda 설치 경로가 다를 경우 수정 필요)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

# 2. CUDA 및 파이썬 관련 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1 # A100에서 P2P 통신 에러 방지용 (필요 시 사용)

# 3. 모델 및 평가 경로 설정
NUM_GPUS=4
MODEL_DIR=./models/DeepSeek-R1-Distill-Qwen-1_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k
OUTPUT_DIR=./eval_results

# 4. 모델 인자 설정 
# OOM 방지를 위해 max_model_length를 8192로 줄였습니다. math_500 테스트에는 충분합니다.
MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

echo "Starting evaluation for: $MODEL_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# 5. LightEval 실행
# slurm 환경이므로 accelerate launch를 통해 띄우는 것이 안전할 수도 있지만,
# lighteval accelerate 커맨드가 내부적으로 처리를 하므로 그대로 유지합니다.
lighteval accelerate "$MODEL_ARGS" "lighteval|math_500|0|0" \
  --output-dir "$OUTPUT_DIR" \
  --save-details

echo "Evaluation finished."