import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# Add the path to the LLM_MMR folder to sys.path
sys.path.append(os.path.abspath('../LLM_MMR'))
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils.constants import openai_key, predict_system_message, get_mlp_layers
from llm.llm import OpenAILLM
import pandas as pd
from utils.templates import *

def replace(args):
  target_model = AutoModelForCausalLM.from_pretrained(
                args.target_model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).half().cuda()
  if args.update_layer:
    original_model = AutoModelForCausalLM.from_pretrained(
                args.aligement_model,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).half().cuda()
    if args.print_layer:
      print("---------------Layers for target_model------------------")
      for i, layer in enumerate(target_model.model.layers):
        print(f"Layer {i}, MLP: {layer.mlp}")
        break
      
      print("---------------Layers for original_model------------------")
      for i, layer in enumerate(original_model.model.layers):
        print(f"Layer {i}, MLP: {layer.mlp}")
        break
        
    print("---------------Updating the model------------------")
    if 'Llama-2' in args.aligement_model or 'vicuna' in args.aligement_model:    
      num_model_layers = len(original_model.model.layers)
      layers = get_mlp_layers(args.aligement_model)
      print(layers)
      for i in layers:
        original_lm_head_weights = original_model.model.layers[i].mlp
        target_model.model.layers[i].mlp = original_lm_head_weights
    # elif ''
      print(target_model.model.layers[i].mlp==original_lm_head_weights)
      
    del original_model
    
  origin_question = pd.read_csv(args.dataset_path)['text'].tolist()[args.index]
  
  if args.prompt:
    template = get_template(args.target_model)
    question = template['prompt'].format(instruction=origin_question)
  else: 
    question = origin_question + " Show here"
  print("The question is: \n", question)

  
  # question = origin_question

  predictor = OpenAILLM(args.model_path, args.openai_key, system_message=predict_system_message.format(origin_question=origin_question))
  # huggingface generate 
  tokenizer = AutoTokenizer.from_pretrained(args.target_model)

  inputs = tokenizer(question, return_tensors="pt")
  inputs = inputs.to('cuda')  # Move the inputs to GPU
  # Generate
  # changed max_token to max_new_token to better control output token length
  generate_ids = target_model.generate(inputs.input_ids, max_new_tokens = 150)
  response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  if args.predict:
    result = predictor.predict([response], origin_question)[0]
  else: 
    result = 0
    print("The response is :\n", response)
  return response, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=1, help='The index of the question')
    parser.add_argument('--no_update_layer', action='store_false', dest='update_layer', help='do not update the layer of the target_model')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='Tap-M/Luna-AI-Llama2-Uncensored',
                        help='The target unaligement model, openai model or open-sourced LLMs')
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../Dataset/harmful.csv')
    parser.add_argument("--eos_num", type=int, default=10)
    parser.add_argument('--output_dict', type=str, default= './Results2/')
    parser.add_argument('--aligement_model', type=str, default='lmsys/vicuna-13b-v1.3',
                        help='The aligement model, openai model or open-sourced LLMs')
    parser.add_argument('--predict', action='store_true', default=False)
    parser.add_argument('--print_layer', action = 'store_true', default=False)
    parser.add_argument('--prompt', action = 'store_true', default=False, help='Use the model prompt for the question')

    args = parser.parse_args()
    args.openai_key = openai_key
    response, result = replace(args)

    df = pd.DataFrame({'Success': [result], 'Response': [response]}, index=[0])

    # save the optim prompts into a csv file
    save_path = args.output_dict + f'{args.target_model}/GPTFuzzer/{args.index}.csv'
    if args.prompt:
          save_path = args.output_dict + f'{args.target_model}/GPTFuzzer/PROMPT_{args.index}.csv'
    
    print("The save path is: ", save_path)
    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)

    