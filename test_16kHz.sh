#! /bin/bash

LOG_INDEX=2

# SAMPLE_RATES=(8000 12000 16000 24000) # UNIVERSAL TEST
SAMPLE_RATES=(4000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python3 main.py \
        --cfg configs/vm_asr_16k_MPD.yaml \
        --resume logs/logs_${LOG_INDEX}/Ultra_Light/16k_FullData_MPD \
        --eval \
        --tag ${SR}_16000 \
        --disable_amp
done