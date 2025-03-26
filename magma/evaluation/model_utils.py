import os

def get_lora_path(lora_base_path):
    """
    Return the path of the first checkpoint found in lora_base_path.
    If none is found, return None.
    """
    if not os.path.exists(lora_base_path):
        return None
    for item in os.listdir(lora_base_path):
        if "checkpoint-" in item:
            return os.path.join(lora_base_path, item)
    return None

def model_specific_prompt(datapoint, llm_name, tokenizer):
    """
    Return a prompt string formatted for the given model.
    For foundation models, return the first element.
    For others (e.g. Llama), use the tokenizer's chat template.
    """
    if llm_name.lower() in ['gpt-4o', 'o1', 'o1-mini', 'o3-mini', 'deepseek-reasoner']:
        return datapoint[0]
    elif any(x in llm_name.lower() for x in ['llama', 'alpaca', 'mistral']):
        return tokenizer.apply_chat_template(datapoint, tokenize=False, add_generation_prompt=False)
    else:
        return datapoint[0]