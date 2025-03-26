import os
import torch
from openai import OpenAI, AzureOpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_utils import get_lora_path
from magma.globals import FOUNDATION_MODELS

def load_inference_client(config):
    llm_name = config["model_name"]
    lora_base_dir = config["lora_base_dir"]
    target_task = config["target_task"]
    graph_size = f"graph_size_{config['graph_size']}" if config["graph_size"] != "ood" else config["graph_size"]
    chat_type = config["chat_type"]
    reasoning_strategy = config["reasoning_strategy"]

    # Build the path to the LoRA checkpoints (if any)
    lora_base_path = os.path.join(lora_base_dir, target_task, graph_size, "llm_data", chat_type, reasoning_strategy)
    lora_path = get_lora_path(lora_base_path)
    
    client = None
    model = None
    tokenizer = None

    if llm_name in FOUNDATION_MODELS:
        # Use OpenAI-based inference
        if config.get("openai_engine", "openai") == "azure":
            print(f'Using Azure OpenAI: {llm_name}')
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                timeout=200
            )
        else:
            print(f'Using OpenAI: {llm_name}')
            client = OpenAI(timeout=200)
    else:
        # Use Hugging Face-based inference
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="auto")
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)
    return client, model, tokenizer, lora_path