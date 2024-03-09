LLAMA2_PROMPT = {
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}


Alpaca_PROMPT = {
    "prompt": '''[INST] <<SYS>>
    You are a helpful assistant.
<</SYS>>

{instruction} [/INST] '''
}


Luna_PROMPT = {
    "prompt": '''USER: {instruction}
ASSISTANT:'''
}

tulu_PROMPT = {
    "prompt": '''<|user|>
{instruction}
<|assistant|>
'''
}

vicuna_PROMPT = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:
    '''
}

WizardLM_PROMPT = {
    "prompt": '''### Human: {instruction}
### Assistant:'''
}

MPT_PROMPT = {
    "prompt": '''A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
### Human: {instruction}
### Assistant:
'''
}


def get_template(name):
    if 'Llama-2' in name:
        return LLAMA2_PROMPT
    elif 'alpaca' in name: # llama2: hfl/chinese-alpaca-2-7b, llama1: chavinlo/alpaca-native
        return Alpaca_PROMPT
    elif 'Luna' in name:
        return Luna_PROMPT
    elif 'tulu' in name:
        return tulu_PROMPT
    elif 'vicuna' in name: # llama2: lmsys/vicuna-13b-v1.5, llama1: lmsys/vicuna-13b-v1.3
        return vicuna_PROMPT
    elif 'WizardLM' in name: # llama2: WizardLM/WizardLM-13B-V1.2, llama1: cognitivecomputations/WizardLM-7B-V1.0-Uncensored
        return WizardLM_PROMPT
    elif 'mpt' in name:
        return MPT_PROMPT
    else:
        return None