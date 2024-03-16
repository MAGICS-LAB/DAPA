#!/bin/bash --login

source activate base
conda activate /home/user/miniconda3/envs/jailbreak
PYTHON_SCRIPT="./plugin_aligner/replace.py"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
DATASET_PATH="./Dataset/harmful.csv"
ADD_EOS=False
# Set the log path based on ADD_EOS
if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer_eos"
else
    LOG_PATH="normal_Logs/${MODEL_PATH}/GPTFuzzer"
fi

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi


# Determine the length of harmful.csv
LENGTH=$(wc -l < $DATASET_PATH)

# Run replace.py for each index from 0 to LENGTH - 1
for (( index=0; index<LENGTH-1; index++ )); do
    /home/user/miniconda3/envs/jailbreak/bin/python3.9 $PYTHON_SCRIPT --alignment_model $MODEL_PATH --target_model $MODEL_PATH --dataset_path $DATASET_PATH --no_update_layer --prompt --test_alignment --predict --index $index > "${LOG_PATH}/${index}.log" 2>&1 &
    sleep 25
done

echo "Finished Running run_jailbreak.sh"
# Wait for all background jobs to finish
wait