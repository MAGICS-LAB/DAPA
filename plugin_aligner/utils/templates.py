LLAMA2_CHAT_PROMPT = {
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''
}

def get_templates(model_path, func):
    if 'Llama-2' in model_path:
        if func == 'no_sys':
            return LLAMA2_PROMPT_no_sys
        elif func == 'GCG':
            return LLAMA2_PROMPT_GCG
        elif func == 'chat':
            return LLAMA2_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "no_sys", "GCG", "chat"')
    elif 'mpt' in model_path:
        if func == 'no_sys':
            return MPT_7B_PROMPT_no_sys
        elif func == 'GCG':
            return MPT_7B_PROMPT_GCG
        elif func == 'chat':
            return MPT_7B_PROMPT
        else:
            raise ValueError(f'Unknown function {func}, should be one of "no_sys", "GCG", "chat"')
    elif 'gemma' in model_path:
        if func == 'GCG':
            return GEMMA_7B_PROMPT_GCG
        elif func == 'chat' or func == 'no_sys':
            return GEMMA_7B_PROMPT
    elif 'Qwen' in model_path:
        if func == 'GCG':
            return QWEN_7B_PROMPT_GCG
        elif func == 'chat':
            return QWEN_7B_PROMPT
    elif 'tulu' in model_path:
        if func == 'GCG':
            return TULU_7B_PROMPT_GCG
        elif func == 'chat':
            return TULU_7B_PROMPT
    elif 'mistral' in model_path:
        if func == 'GCG':
            return MISTRAL_7B_PROMPT_GCG
        elif func == 'chat':
            return MISTRAL_7B_PROMPT
    elif 'vicuna' in model_path:
        if func == 'GCG':
            return VICUNA_7B_PROMPT_GCG
        elif func == 'chat':
            return VICUNA_7B_PROMPT
    else:
        raise ValueError(f'Unknown model {model_path}, should be one of "Llama-2", "mpt"')