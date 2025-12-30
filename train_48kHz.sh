#! /bin/bash

LOG_INDEX=9

python3 main.py \
    --cfg configs/upvm_asr_48k_MPD.yaml \
    --tag 48k_8k_FullData_MPD \
    --output logs/logs_${LOG_INDEX}/ \
    --input_sr 16000 \
    --disable_amp \
    --resume logs/logs_${LOG_INDEX}/Ultra_Light/48k_FullData_MPD

# UNIVERSAL VERSION
# python3 main.py \
#     --cfg configs/upvm_asr_48k_MPD.yaml \
#     --tag 48k_8k_FullData_MPD \
#     --output logs/logs_${LOG_INDEX}/ \
#     --input_sr 16000 \
#     --disable_amp \
#     --resume logs/logs_${LOG_INDEX}/Ultra_Light/48k_FullData_MPD