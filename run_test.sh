#!/bin/bash

# source activate base
# conda activate jailbreak
PYTHON_SCRIPT="./plugin_aligner/replace.py"
# MODEL_PATH="lmsys/vicuna-13b-v1.5"
DATASET_PATH="./Dataset/harmful.csv"
ADD_EOS=False
# # Set the log path based on ADD_EOS
# if [ "$ADD_EOS" = "True" ]; then
#     LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer_eos"
# else
#     LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer"
# fi

# # Create the log directory if it does not exist
# mkdir -p "$LOG_PATH"

export HF_TOKEN='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'
export HF_HOME="/projects/p32013/.cache/"

# Function to find the first available GPU
# find_free_gpu() {
#     for i in {0..3}; do
#         free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
#         if [ "$free_mem" -ge 80000 ]; then
#             echo $i
#             return
#         fi
#     done

#     echo "-1" # Return -1 if no suitable GPU is found
# }

find_free_gpu() {
    for i in {0..1}; do
        if nvidia-smi -i $i | grep 'No running processes found' > /dev/null; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no free GPU is found
}


# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi


# Determine the length of harmful.csv
LENGTH=$(wc -l < $DATASET_PATH)
echo "Start Running run_jailbreak.sh for model: ${MODEL_PATH}"
# Run replace.py for each index from 0 to LENGTH - 1
ali_models="meta-llama/Llama-2-13b-chat-hf"
target_models=("meta-llama/Llama-2-13b-hf")
indices=(0 1 2 4 5)
for TARGET_MODEL in "${target_models[@]}"
do
    echo "Target model: ${TARGET_MODEL}"
    # Set the log path based on ADD_EOS
    if [ "$ADD_EOS" = "True" ]; then
        LOG_PATH="update_Logs/${TARGET_MODEL}/GPTFuzzer_eos"
    else
        LOG_PATH="update_Logs/${TARGET_MODEL}/GPTFuzzer"
    fi

    # Create the log directory if it does not exist
    mkdir -p "$LOG_PATH"
    for index in "${indices[@]}"; do
        FREE_GPU=1

        CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --alignment_model $ali_models --target_model $TARGET_MODEL --dataset_path $DATASET_PATH --prompt --predict --index $index --output_dict './result_update_layer/' > "${LOG_PATH}/${index}.log" 2>&1 
        echo "Task $index on GPU $FREE_GPU finished."

        sleep 25 # Wait for 25 seconds before starting the next task
    done
done

ANALYZE_SCRIPT="./plugin_aligner/analyze.py"
DIR_PATH="./result_update_layer/meta-llama/Llama-2-13b-hf/GPTFuzzer"
/home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $ANALYZE_SCRIPT --directory_path $DIR_PATH
echo "Finished Running run_jailbreak.sh"
# Wait for all background jobs to finish
wait
