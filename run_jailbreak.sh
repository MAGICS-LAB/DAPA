#!/bin/bash

# source activate base
# conda activate jailbreak
PYTHON_SCRIPT="./plugin_aligner/replace.py"
MODEL_PATH="lmsys/vicuna-13b-v1.5"
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
# export HF_HOME="/projects/p32013/.cache/"

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
echo "Start Running run_jailbreak.sh for model: $MODEL_PATH"
# Run replace.py for each index from 0 to LENGTH - 1
models=("google/gemma-2b" "google/gemma-7b")
for MODEL_PATH in "${models[@]}"
do
    echo "Running model: $MODEL_PATH"
    # Set the log path based on ADD_EOS
    if [ "$ADD_EOS" = "True" ]; then
        LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer_eos"
    else
        LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer"
    fi

    # Create the log directory if it does not exist
    mkdir -p "$LOG_PATH"
    for (( index=0; index<LENGTH-1; index++ )); do
        FREE_GPU=2

        # # Keep looping until a free GPU is found
        # while [ $FREE_GPU -eq -1 ]; do
        #     FREE_GPU=$(find_free_gpu)
        #     if [ $FREE_GPU -eq -1 ]; then
        #         sleep 5 # Wait for 5 seconds before trying to find a free GPU again
        #     fi
        # done

        # (
        #     CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --alignment_model $MODEL_PATH --target_model $MODEL_PATH --dataset_path $DATASET_PATH --no_update_layer --prompt --predict --index $index --output_dict './result_unalignt' > "${LOG_PATH}/${index}.log" 2>&1 &
        #     echo "Task $index on GPU $FREE_GPU finished."
        # ) &
        CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --alignment_model $MODEL_PATH --target_model $MODEL_PATH --dataset_path $DATASET_PATH --no_update_layer --prompt --predict --index $index --output_dict './result_unalignt/' > "${LOG_PATH}/${index}.log" 2>&1 
        echo "Task $index on GPU $FREE_GPU finished."
    (CUDA_VISIBLE_DEVICES=$FREE_GPU python $PYTHON_SCRIPT --alignment_model $MODEL_PATH --target_model $MODEL_PATH --dataset_path $DATASET_PATH --no_update_layer --prompt --test_alignment --predict --index $index > "${LOG_PATH}/${index}.log" 2>&1 &
    echo "Task $index on GPU $FREE_GPU finished."
    ) &

        sleep 25 # Wait for 25 seconds before starting the next task
    done
done

echo "Finished Running run_jailbreak.sh"
# Wait for all background jobs to finish
wait
