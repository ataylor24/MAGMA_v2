import os
import json
import argparse
import yaml
from datasets import load_from_disk
from evaluation.engine import load_inference_client
from evaluation.inference import run_inference

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a given task using a YAML configuration."
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="Path to the YAML configuration file.")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up inference client, model, and tokenizer (if applicable)
    client, model, tokenizer, lora_path = load_inference_client(config)
    
    # Construct the evaluation data path from config
    evaldata_save_path = config["prompt_data"]
    print(f'Loading evaluation data from: {evaldata_save_path}')
    evaldata = load_from_disk(evaldata_save_path)
    
    # Run inference
    results = run_inference(evaldata, config, client, model, tokenizer)
    
    # Save results to a JSON file
    save_path = config["output_json"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print("Results saved to", save_path)

if __name__ == "__main__":
    main()