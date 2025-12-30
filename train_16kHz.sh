#! /bin/bash

LOG_INDEX=2

python3 main.py \
    --cfg configs/upvm_asr_16k_MPD.yaml \
    --tag 16k_4k_FullData_MPD \
    --output logs/logs_${LOG_INDEX}/ \
    --input_sr 4000 \
    --disable_amp \
    --resume logs/logs_${LOG_INDEX}/Ultra_Light/16k_FullData_MPD

# Versatile version
# python3 main.py \
#     --cfg configs/vm_asr_16k_MPD.yaml \
#     --tag 16k_4k_FullData_MPD \
#     --output logs/logs_${LOG_INDEX}/ \
#     --disable_amp \
#     --resume logs/logs_${LOG_INDEX}/Ultra_Light/16k_FullData_MPD
