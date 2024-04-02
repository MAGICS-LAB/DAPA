import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Add the path to the LLM_MMR folder to sys.path
sys.path.append(os.path.abspath('../LLM_MMR'))
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.constants import openai_key, predict_system_message, get_mlp_layers
from llm.llm import OpenAILLM
import pandas as pd
from utils.templates import *
# need to include import hf_olmo to run OLMO model
import hf_olmo

os.environ['HF_HOME'] = "/projects/p32013/.cache/"

        
def replace(args):
  target_model = AutoModelForCausalLM.from_pretrained(
                args.target_model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                # device_map='auto',
            ).cuda()
  
  if args.update_layer:
    original_model = AutoModelForCausalLM.from_pretrained(
                args.alignment_model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(target_model.dtype).cuda()
    
    if args.print_model:
      print("---------------target_model------------------")
      print(target_model.__dict__)
      
      print("---------------original_model------------------")
      print(original_model.__dict__)
        
    print("---------------Updating the model------------------")

    model_layers_type = ['Llama-2','vicuna','Mistral','Qwen1.5','Yi','deepseek','gemma']
    transformer_h_type = ['falcon','Qwen']
    transformer_blocks_type = ['mpt']
    if any(ele in args.alignment_model for ele in model_layers_type):
      num_model_layers = len(original_model.model.layers)
      layers = get_mlp_layers(args.alignment_model)
      print("The layers are: ", layers)
      for i in layers:
        original_lm_head_weights = original_model.model.layers[i].mlp
        target_model.model.layers[i].mlp = original_lm_head_weights
        
    elif any(ele in args.alignment_model for ele in transformer_h_type):
      num_model_layers = len(original_model.transformer.h)
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.transformer.h[i].mlp
        target_model.transformer.h[i].mlp = original_lm_head_weights
        
    elif any(ele in args.alignment_model for ele in transformer_blocks_type):
      num_model_layers = len(original_model.transformer.blocks)
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.transformer.blocks[i].ffn
        target_model.transformer.blocks[i].ffn = original_lm_head_weights
    
    elif 'OLMo' in args.alignment_model:
      num_model_layers = len(original_model.model.transformer.blocks)
      layers = get_mlp_layers(args.alignment_model)
      for i in layers:
        original_lm_head_weights = original_model.model.transformer.blocks[i].ff_out
        target_model.model.transformer.blocks[i].ff_out = original_lm_head_weights
      
    del original_model

  origin_question = pd.read_csv(args.dataset_path)['text'].tolist()[args.index]
  
  if args.prompt:
    if args.test_alignment:
      template = get_chat_template(args.target_model)
      question = template['prompt'].format(instruction=origin_question)
    else:
      template = get_template(args.target_model)
      if template:
        question = template['prompt'].format(instruction=origin_question)
      else: 
        question = origin_question + ". Sure,"
  else:
    question = origin_question
  print("The question is: \n", question)

  predictor = OpenAILLM(args.model_path, args.openai_key, system_message=predict_system_message.format(origin_question=origin_question))
  # huggingface generate 
  
  tokenizer = AutoTokenizer.from_pretrained(args.target_model,  trust_remote_code=True)

  inputs = tokenizer(question, return_tensors="pt")
  inputs = inputs.to('cuda')  # Move the inputs to GPU
  # Generate
  # changed max_token to max_new_token to better control output token length
  generate_ids = target_model.generate(inputs.input_ids, max_new_tokens = 350)
  response = tokenizer.batch_decode(generate_ids[:,len(inputs.input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  #response = response.replace(question,"")
  print("The response is :\n", response)
  if args.predict:
    result = predictor.predict([response], origin_question)[0]
    print("ChatGPT's result is ", result)
  else: 
    result = 0
  
  return question, response, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=0, help='The index of the question')
    parser.add_argument('--no_update_layer', action='store_false', dest='update_layer', help='do not update the layer of the target_model')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='Tap-M/Luna-AI-Llama2-Uncensored',
                        help='The target unaligement model, openai model or open-sourced LLMs')
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../Dataset/harmful.csv')
    parser.add_argument('--test_alignment', action='store_true', default=False)
    parser.add_argument("--eos_num", type=int, default=10)
    parser.add_argument('--output_dict', type=str, default= './Results2/')
    parser.add_argument('--alignment_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='The aligement model, openai model or open-sourced LLMs')
    parser.add_argument('--predict', action='store_true', default=False)
    parser.add_argument('--print_model', action = 'store_true', default=False)
    parser.add_argument('--prompt', action = 'store_true', default=False, help='Use the model prompt for the question')

    args = parser.parse_args()
    args.openai_key = openai_key
    print(f"Alignment model: {args.alignment_model}, Target model: {args.target_model}")
    question, response, result = replace(args)

    df = pd.DataFrame({'Question':[question], 'Response': [response], 'Success': [result]}, index=[0])

    # save the optim prompts into a csv file
    save_path = args.output_dict + f'{args.target_model}/GPTFuzzer/{args.index}.csv'
    if args.prompt:
      save_path = args.output_dict + f'{args.target_model}/GPTFuzzer/PROMPT_{args.index}.csv'
    
    print("The save path is: ", save_path)
    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path, index=False)

    