LLAMA2_PROMPT = {
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
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


def get_template(name):
    if 'Llama-2' in name:
        return LLAMA2_PROMPT
    elif 'alpaca' in name:
        return Alpaca_PROMPT
    elif 'Luna' in name:
        return Luna_PROMPT
    elif 'tulu' in name:
        return tulu_PROMPT
    elif 'vicuna' in name:
        return vicuna_PROMPT
    else:
        return None