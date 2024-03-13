LLAMA2_PROMPT_no_sys = {
    "description": "Llama 2 prompt without system message",
    "prompt": '''[INST] <<SYS>>

<</SYS>>

{instruction} [/INST] '''
}

Alpaca_PROMPT = {
    "prompt": '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:'''
}

# add an empty line
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
   "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
  
USER: {instruction}
ASSISTANT:'''
}

# revised - (WizardLM-13B-V1.2)
WizardLM_PROMPT = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} Assistant:'''
}

# revised - (WizardLM-7B-V1.0-Uncensored-GGML)
WizardLM_PROMPT_2 = {
    "prompt": '''You are a helpful AI assistant.

USER: {instruction}
ASSISTANT'''
}

# not sure - (see my document)
MPT_PROMPT = {
    "prompt": '''- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.
user: {instruction}
assistant:
'''
}


def get_template(name):
    if 'Llama-2' in name:
        return LLAMA2_PROMPT_no_sys
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