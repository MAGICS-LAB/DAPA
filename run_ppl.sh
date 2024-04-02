#!/bin/bash

PYTHON_SCRIPT="./plugin_aligner/evulate.py"

NO_UPDATE_LAYER=false

# Check if the third argument is --no_update_layer
if [[ "$1" == "--no_update_layer" ]]; then
    NO_UPDATE_LAYER=true
fi

echo "NO_UPDATE_LAYER is set to $NO_UPDATE_LAYER"

export HF_TOKEN='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'
export HF_HOME="/projects/p32013/.cache/"


ali_models="mistralai/Mistral-7B-Instruct-v0.2"
target_models=("cognitivecomputations/dolphin-2.2.1-mistral-7b" "HuggingFaceH4/zephyr-7b-alpha" "cognitivecomputations/dolphin-2.6-mistral-7b-dpo" "abhishekchohan/mistral-7B-forest-dpo")
# ("mistralai/Mistral-7B-v0.1" "teknium/OpenHermes-2-Mistral-7B" "cognitivecomputations/dolphin-2.2.1-mistral-7b" "HuggingFaceH4/zephyr-7b-alpha" "cognitivecomputations/dolphin-2.6-mistral-7b-dpo" "abhishekchohan/mistral-7B-forest-dpo")

# Set the log path based on ADD_EOS
if $NO_UPDATE_LAYER; then
    LOG_PATH="ppl_Logs/unaligned"
else
    LOG_PATH="ppl_Logs/update"
fi

mkdir -p "$LOG_PATH"

for TARGET_MODEL in "${target_models[@]}"
do
    FREE_GPU=0
    if $NO_UPDATE_LAYER; then
        CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --type 'ppl' --model $TARGET_MODEL --alignment_model $ali_models --no_update_layer> "${LOG_PATH}/temp.log" 2>&1 
    else
        CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --type 'ppl' --model $TARGET_MODEL --alignment_model $ali_models> "${LOG_PATH}/temp.log" 2>&1 
    fi
    sleep 25 # Wait for 25 seconds before starting the next task
    tail -n 1 "${LOG_PATH}/temp.log"
    
done

wait