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

mistral_chat_PROMPT = { "prompt": '''[INST] {instruction} [/INST]'''}





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

# add an empty line
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


Redmond_Puffin_13B_PROMPT = {
    "prompt": '''USER: {instruction}
    
ASSISTANT:'''
}

Chinese_alpaca_2_7b_PROMPT = {
    "[INST] <<SYS>>\n"
    "{You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
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

h2ogpt_PROMPT = {
    "prompt": '''<|prompt|>{instruction}<|endoftext|><|answer|>'''}


WizardLM_10_PROMPT = {
    "prompt": '''You are a helpful AI assistant.

    USER: <instruction>
    ASSISTANT:
    '''
}

falcon_sft_PROMPT = {
    "prompt": '''<|prompter|>{instruction}<|endoftext|><|assistant|>'''
}

openhermes_mistral_prompt = {
    'prompt': '''<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

dolphin_prompt = { "prompt": '''<|im_start|>system
You are Dolphin, a helpful AI assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

openchat_prompt = { "prompt": '''GPT4 Correct User: {instruction}<|end_of_turn|>GPT4 Correct Assistant:'''}

Openorca_prompt = { "prompt": '''<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!
<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''}


Starling_prompt = { "prompt": '''GPT4 Correct User: {instruction}<|end_of_turn|>GPT4 Correct Assistant:'''}







#https://huggingface.co/Syed-Hasan-8503/openhermes-gemma-2b-it
openhermes_gemma_prompt = {"prompt": '''<start_of_turn>user
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
    elif 'chinese-alpaca' in name:
        return Chinese_alpaca_2_7b_PROMPT
    elif 'Redmond-Puffin' in name:
        return Redmond_Puffin_13B_PROMPT
    elif 'alpaca' in name or 'Gemmalpaca' in name: # llama2: hfl/chinese-alpaca-2-7b, llama1: chavinlo/alpaca-native
        return Alpaca_PROMPT
    elif 'falcon' in name and 'sft' in name:
        return falcon_sft_PROMPT
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
    elif 'openhermes' in name and 'gemma' in name:
        return openhermes_gemma_prompt
    elif 'OpenHermes' in name and 'Mistral' in name:
        return openhermes_mistral_prompt
    elif 'deepseek' in name and 'chat' in name:
        return deepseek_chat_prompt
    elif 'OpenOrca' in name:
        return Openorca_prompt
    elif 'deepseek' in name:
        return deepseek_prompt
    elif 'Wukong' in name:
        return Wukong_prompt
    elif 'zephyr' in name:
        return zephyr_prompt
    elif 'h2ogpt' in name:
        return h2ogpt_PROMPT
    elif 'dolphin' in name:
        return dolphin_prompt
    elif 'openchat' in name:
        return openchat_prompt
    elif 'Starling' in name:
        return Starling_prompt
    else:
        return None