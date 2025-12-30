#! /bin/bash

LOG_INDEX=9

# SAMPLE_RATES=(8000 12000 16000 24000) # UNIVERSAL TEST
SAMPLE_RATES=(16000)

# Loop over sample rates and run the Python script
for SR in "${SAMPLE_RATES[@]}"
do
    python3 main.py \
        --cfg configs/vm_asr_48k_MPD.yaml \
        --resume logs/logs_${LOG_INDEX}/Ultra_Light/48k_FullData_MPD \
        --eval \
        --tag ${SR}_48000 \
        --disable_amp
done