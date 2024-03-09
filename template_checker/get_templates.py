from fastchat.model import get_conversation_template

def chat_templates(model_path, func):
    conv_temp = get_conversation_template(model_path)
    # monkey patch for Llama-2 system message
    if 'Llama-2' in model_path:
        conv_temp.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
    prompt = '{instruction}'
    if func=='no_sys':
        # chat template without system message
        conv_temp.system_message = None
        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)
        return conv_temp.get_prompt()
    elif func=='GCG':
        # system template only, used for GCG
        return conv_temp.get_prompt()
    elif func=='chat':
        # chat template with system message
        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)
        return conv_temp.get_prompt()
    else:
        raise ValueError(f'Unknown function {func}, should be one of "no_sys", "GCG", "chat"')
    
    
if __name__ == "__main__":
    template = chat_templates('mosaicml/mpt-7b-storywriter', 'chat')
    print(template)
    print("Converting new line to \\n")
    print(template.replace('\n', '\\n'))
        