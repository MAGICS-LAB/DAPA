#!/bin/bash
#SBATCH -A p32013 ## Required: your allocation/account name, i.e. eXXXX, pXXXX or bXXXX
#SBATCH -p gengpu ## Required: (buyin, short, normal, long, gengpu, genhimem, etc)
#SBATCH --gres gpu:a100:2
#SBATCH --constraint=sxm
#SBATCH -t 48:00:00 ## Required: How long will the job need to run (remember different partitions have restrictions on this parameter)
#SBATCH --nodes 1 ## how many computers/nodes do you need (no default)
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 2 ## how many cpus or processors do you need on per computer/node (default value 1)
#SBATCH --mem 160G ## how much RAM do you need per computer/node (this affects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=robin ## When you run squeue -u 
#SBATCH --mail-type END ## BEGIN, END, FAIL or ALL
#SBATCH --mail-user=xxxx
#SBATCH --output=robin/output1.out
#SBATCH --error=robin/error1.err

module purge all
module load python-miniconda3/4.12.0
module load cuda/cuda-12.1.0-openmpi-4.1.4
source activate /projects/intro/envs/pytorch-1.11-py38
#conda create -n jailbreak python=3.9
conda activate jailbreak





export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd /projects/p32013/XLLM

PYTHON_SCRIPT="./Experiments/normal_exp.py"
MODEL_PATH="allenai/tulu-2-dpo-7b"
ADD_EOS=False
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_TOKEN='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'
export HF_HOME="/projects/p32013/.cache/"


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

# Function to find the first available GPU
find_free_gpu() {
    for i in {0..1}; do
        if nvidia-smi -i $i | grep 'No running processes found' > /dev/null; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no free GPU is found
}

# Start the jobs with GPU assignment
for index in {0..1}; do
    FREE_GPU=-1

    # Keep looping until a free GPU is found
    while [ $FREE_GPU -eq -1 ]; do
        FREE_GPU=$(find_free_gpu)
        if [ $FREE_GPU -eq -1 ]; then
            sleep 5 # Wait for 5 seconds before trying to find a free GPU again
        fi
    done

    # Run the Python script on the free GPU
    (
        CUDA_VISIBLE_DEVICES=$FREE_GPU /home/hlv8980/.conda/envs/jailbreak/bin/python3.9 -u "$PYTHON_SCRIPT" --index $index --target_model $MODEL_PATH $ADD_EOS_FLAG > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU $FREE_GPU finished."
    ) &

    # Wait for 30 seconds to give the GPU some time to allocate memory
    sleep 30
done

# Wait for all background jobs to finish
wait
