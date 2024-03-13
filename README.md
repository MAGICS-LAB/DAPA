## Install Command
conda create --name jailbreak python=3.9.18
pip3 install torch torchvision torchaudio
python -m pip install transformers
python -m pip install openai
python -m pip install pandas
python -m pip install einops
python -m pip install accelerate
python -m pip install sentencepiece
python -m pip install protobuf
python -m pip install transformers_stream_generator
python -m pip install tiktoken
python -m pip install ai2-olmo
python -m pip install autoawq
python -m pip install importlib-metadata
python -m pip install auto-gptq
python -m pip install flash_attn


export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
export PATH=/usr/local/cuda/bin:/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:/home/hlv8980/.local/bin:/home/hlv8980/google-cloud-sdk/bin:/home/hlv8980/anaconda3/envs/jailbreak/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin


## Run Command
python replace.py --target_model meta-llama/Llama-2-7b-hf
