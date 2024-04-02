from evaluate import load
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from utils.templates import *
import os
from tqdm import tqdm
from utils.constants import openai_key, predict_system_message, get_mlp_layers

seed_value = 42  # Example seed value

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)  # For all CUDA devices
    
os.environ['HF_HOME'] = "/projects/p32013/.cache/"


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument('--alignment_model', type=str, default='google/gemma-2b-it',
                        help='The aligement model, openai model or open-sourced LLMs')
parser.add_argument('--no_update_layer', action='store_false', dest='update_layer', help='do not update the layer of the target_model')

args = parser.parse_args()


def replace(model, args):
  if args.update_layer:
    original_model = AutoModelForCausalLM.from_pretrained(
                args.alignment_model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(model.dtype).cuda()
        
    print("---------------Updating the model------------------")

    model_layers_type = ['Llama-2','vicuna','Mistral','Qwen1.5','Yi','deepseek','gemma']
    transformer_h_type = ['falcon','Qwen']
    transformer_blocks_type = ['mpt']
    if any(ele in args.alignment_model for ele in model_layers_type):
      layers = get_mlp_layers(args.alignment_model)
      print("The layers are: ", layers)
      for i in layers:
        original_lm_head_weights = original_model.model.layers[i].mlp
        model.model.layers[i].mlp = original_lm_head_weights
        
    elif any(ele in args.alignment_model for ele in transformer_h_type):
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.transformer.h[i].mlp
        model.transformer.h[i].mlp = original_lm_head_weights
        
    elif any(ele in args.alignment_model for ele in transformer_blocks_type):
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.transformer.blocks[i].ffn
        model.transformer.blocks[i].ffn = original_lm_head_weights
    
    elif 'OLMo' in args.alignment_model:
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.model.transformer.blocks[i].ff_out
        model.model.transformer.blocks[i].ff_out = original_lm_head_weights
      
    del original_model
  return model


if args.type == "bleu":
  bleu = load("bleu")
  predictions = pd.read_csv("predictions.csv")
  references = pd.read_csv("references.csv")
  # predictions = ["hello there general kenobi", "foo bar foobar"]
  # references = [
  #   ["hello there general kenobi"],
  #   ["foo bar foobar"]]
  results = bleu.compute(predictions=predictions, references=references)
  print('bleu:', results['bleu'])

elif args.type == "ppl":
  model = AutoModelForCausalLM.from_pretrained(
                args.model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16).cuda()
  tokenizer = AutoTokenizer.from_pretrained(args.model,  trust_remote_code=True)
  model = replace(model, args)
  # Load WikiText-2 dataset
  dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')
  encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
  
  max_length = model.config.max_position_embeddings   #may need change, check cofig file
  stride = 512
  seq_len = encodings.input_ids.size(1)

  nlls = []
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride)):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)

          # loss is calculated using CrossEntropyLoss which averages over valid labels
          # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
          # to the left by 1.
          neg_log_likelihood = outputs.loss

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break

  ppl = torch.exp(torch.stack(nlls).mean())
  if args.update_layer:
    print(f'The updated model \"{args.model}\" prepexity is: {ppl}')
  else:
    print(f'The unaligned model \"{args.model}\" prepexity is: {ppl}') 


