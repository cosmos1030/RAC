#!/bin/bash

# 1. 할당받은 GPU의 UUID 리스트 추출
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)

# 2. 할당받은 CPU 코어 리스트 추출 
CPULIST=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')

# 3. Docker 실행 (가이드라인의 '"device=..."' 형식을 정확히 맞춤)
# 바깥쪽은 큰따옴표("), 안쪽은 작은따옴표('), 그 안에 다시 큰따옴표(")를 넣어야 
# 쉘이 최종적으로 "device=GPU-xxx,GPU-yyy" (따옴표 포함)를 Docker에게 전달합니다.
docker run -it --rm \
    --name ${USER}_open_r1 \
    --gpus '"device='${UUIDLIST}'"' \
    --cpuset-cpus "${CPULIST}" \
    --shm-size=64gb \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.01-py3 \
    /bin/bash