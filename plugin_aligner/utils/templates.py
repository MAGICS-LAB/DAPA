LLAMA2_chat_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}

Falcon_chat_PROMPT = { 'prompt': '''User: {instruction}

Assistant:
'''}

mistral_chat_PROMPT = { "prompt": '''[INST] {instruction} [/INST]'''}

MPT_chat_PROMPT = {
    "description": "MPT 7B chat one shot prompt",
    "prompt": '''<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

GEMMA_chat_PROMPT = {
    "description": "GEMMA 7B chat one shot prompt",
    "prompt": '''<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
'''
}

QWEN_chat_PROMPT = {
    "description": "Qwen 7B chat prompt",
    "prompt": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

Yi_chat_PROMPT = { "prompt": '''<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''}

deepseek_chat_prompt = { "prompt": '''<｜begin▁of▁sentence｜>User: {instruction}
                   
                   Assistant:
                   '''}


olmo_chat_PROMPT = { "prompt": '''<|user|>
                    {instruction}
                    <|assistant|>
                    '''}


qwen_15_chat_PROMPT = { "prompt": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''}







LLAMA2_PROMPT_no_sys = {
    "description": "Llama 2 prompt without system message",
    "prompt": '''[INST] <<SYS>>

<</SYS>>

{instruction} [/INST] 
'''
}




Alpaca_PROMPT = {
    "prompt": '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''
}


Luna_PROMPT = {
    "prompt": '''USER: {instruction}
ASSISTANT:
'''
}

tulu_PROMPT = {
    "prompt": '''<|user|>
{instruction}
<|assistant|>
'''
}

vicuna_PROMPT = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 
    
    USER: {instruction} 
    ASSISTANT:
    '''
}

WizardLM_12_PROMPT = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 
    
    USER: {instruction} 
    ASSISTANT:
    '''
}

WizardLM_10_PROMPT = {
    "prompt": '''You are a helpful AI assistant.

    USER: <instruction>
    ASSISTANT:
    '''
}

MPT_PROMPT = {
    "prompt": '''A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
### Human: {instruction}
### Assistant:
'''
}


#https://huggingface.co/Syed-Hasan-8503/openhermes-gemma-2b-it
openhermes_prompt = {"prompt": '''<start_of_turn>user
 {instruction}<end_of_turn>
 <start_of_turn>model
 '''}


deepseek_prompt = { "prompt": '''{instruction}. The output is
                   '''
}

Wukong_prompt = { "prompt": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''}


zephyr_prompt = { "prompt": '''<bos><|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''}

def get_template(name):
    if 'Llama-2' in name:
        return LLAMA2_PROMPT_no_sys
    elif 'alpaca' in name or 'Gemmalpaca' in name: # llama2: hfl/chinese-alpaca-2-7b, llama1: chavinlo/alpaca-native
        return Alpaca_PROMPT
    elif 'Luna' in name:
        return Luna_PROMPT
    elif 'tulu' in name:
        return tulu_PROMPT
    elif 'vicuna' in name: # llama2: lmsys/vicuna-13b-v1.5, llama1: lmsys/vicuna-13b-v1.3
        return vicuna_PROMPT
    elif 'WizardLM' in name and 'V1.2' in name: # llama2: WizardLM/WizardLM-13B-V1.2, llama1: cognitivecomputations/WizardLM-7B-V1.0-Uncensored
        return WizardLM_12_PROMPT
    elif 'WizardLM' in name and 'V1.0' in name:
        return WizardLM_10_PROMPT
    elif 'mpt' in name:
        return MPT_PROMPT
    elif 'openhermes' in name:
        return openhermes_prompt
    elif 'deepseek' in name and 'chat' in name:
        return deepseek_chat_prompt
    elif 'deepseek' in name:
        return deepseek_prompt
    elif 'Wukong' in name:
        return Wukong_prompt
    elif 'zephyr' in name:
        return zephyr_prompt
    else:
        return None