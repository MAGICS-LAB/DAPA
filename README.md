# Decoupled Alignment for Robust Plug-and-Play Adaptation

This project implements a decoupled alignment approach for robust plug-and-play adaptation of language models. It provides tools for analyzing and modifying model behavior while maintaining alignment with desired objectives.

## Project Structure

```
.
├── plugin_aligner/       # Core alignment implementation
│   ├── replace.py       # Main replacement logic
│   ├── analyze.py       # Analysis tools
│   ├── evulate.py       # Evaluation utilities
│   └── utils/          # Helper utilities
├── Dataset/             # Dataset directory
├── template_checker/    # Template verification tools
└── scripts
    ├── run_test.sh     # Testing script
    ├── run_jailbreak.sh # Jailbreak testing
    └── run_ppl.sh      # Perplexity evaluation
```

## Installation

1. Create and activate a conda environment:
```bash
conda create --name jailbreak python=3.9.18
conda activate jailbreak
```

2. Install required packages:
```bash
# PyTorch and related packages
pip3 install torch torchvision torchaudio

# Transformers and language model tools
pip install -U transformers
pip install openai pandas einops accelerate
pip install sentencepiece protobuf
pip install transformers_stream_generator tiktoken
pip install ai2-olmo autoawq auto-gptq
pip install sympy importlib-metadata
```

3. Set up CUDA environment (if using GPU):
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PATH=/usr/local/cuda/bin:$PATH
```

## Usage

### Running Tests
To run tests with a specific model:
```bash
python plugin_aligner/replace.py --target_model meta-llama/Llama-2-7b-hf
```

### Available Scripts
- `run_test.sh`: Run comprehensive tests
- `run_jailbreak.sh`: Evaluate model robustness
- `run_ppl.sh`: Calculate perplexity metrics

### Configuration
- Set `HF_TOKEN` environment variable for Hugging Face model access
- Adjust GPU settings in scripts as needed
- Modify dataset paths in configuration files

## License
See LICENSE file for details.
```
