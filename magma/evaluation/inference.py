import copy
from tqdm import tqdm
import torch

def generate_partial_examples(data, llm_name, reasoning_strategy):
    """
    Simplified function to split chat data into an example (input) and ground truth.
    Here we assume each message is a dict with a "content" key.
    """
    examples = []
    ground_truths = []
    for i, msg in enumerate(data):
        if i == 0:
            examples.append(msg["content"])
        else:
            ground_truths.append(msg["content"])
    # In this simple version, we do not use an intermediate prompt.
    intermediate_prompt = None
    return examples, ground_truths, intermediate_prompt

def run_inference(evaldata, config, client, model, tokenizer):
    """
    Run inference on the evaluation data.
    This simplified version iterates over the evaldata, uses generate_partial_examples
    to construct a prompt, and then calls the appropriate client/model.
    """
    results = []
    batch_size = config.get("batch_size", 1)
    llm_name = config["model_name"]
    
    # Loop over the evaluation data in batches
    for i in tqdm(range(0, len(evaldata), batch_size), desc="Running Inference"):
        batch = evaldata[i:i+batch_size] if batch_size > 1 else [evaldata[i]]
        for dp in batch:
            outputs = []
            # Assume dp has a "messages" key containing a list of message dicts.
            examples, ground_truths, _ = generate_partial_examples(dp["messages"], llm_name, config["reasoning_strategy"])
            prompt = examples[0]  # Simplified: using the first message as prompt

            if llm_name in ['gpt-4o', 'o1', 'o1-mini', 'o3-mini', 'deepseek-reasoner']:
                # Use OpenAI client
                message = {"role": "user", "content": prompt}
                response = client.chat.completions.create(model=llm_name, messages=[message])
                for choice in response.choices:
                    outputs.append(choice.message.content)
            else:
                # Use Hugging Face inference (simplified using model.generate)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                output_ids = model.generate(**inputs, max_new_tokens=100)
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                outputs.append(output_text)
                
            results.append({
                "traj_id": dp["traj_id"],
                "input": prompt,
                "ground_truth": ground_truths[0] if ground_truths else "",
                "pred": outputs
            })
    return results