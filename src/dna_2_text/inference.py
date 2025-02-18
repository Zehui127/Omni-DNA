import argparse
import json
import os
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def preprocess_response(response, mask_token="[MASK]"):
    """
    Preprocess the response to extract text after the [MASK] token.

    Args:
        response (str): The raw model output.
        mask_token (str): The token after which the response is extracted.

    Returns:
        str: Processed response text.
    """
    if mask_token in response:
        response = response.split(mask_token, 1)[1]
    response = re.sub(r'^[\sATGC]+', '', response)
    return response

def generate(message, model, tokenizer):
    message = message + "[MASK]"
    tokenized_message = tokenizer(
        [message], return_tensors='pt', return_token_type_ids=False, add_special_tokens=True
    ).to('cuda')
    response = model.generate(**tokenized_message, max_new_tokens=110, do_sample=False)
    reply = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    return preprocess_response(reply)

def main():
    parser = argparse.ArgumentParser(description="Generate responses using a pretrained model and save output to JSON.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated outputs.")
    args = parser.parse_args()

    model_tokenizer_path = "zehui127/Omni-DNA-DNA2Function"
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path).to('cuda')

    hf_dataset_path = "zehui127/Omni-DNA-Dataset-DNA2Text"
    num_inference = 50
    test_dataset = load_dataset(hf_dataset_path)['test']

    if len(test_dataset) > num_inference:
        sampled_dataset = test_dataset.shuffle(seed=42).select(range(num_inference))
    else:
        sampled_dataset = test_dataset

    output_data = []
    for entry in tqdm(sampled_dataset, desc="Generating responses"):
        instruction = entry.get("instruction", "")
        ground_truth = entry.get("output", "")
        if instruction:
            model_output = generate(instruction, model, tokenizer)
            output_data.append({
                "Instruction": instruction,
                "Model_Output": model_output,
                "Ground_Truth": ground_truth
            })
    des_path = os.path.join(args.output_path,"dna2func_samples.json")
    with open(des_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Generated outputs saved to {args.output_path}")

if __name__ == "__main__":
    main()
