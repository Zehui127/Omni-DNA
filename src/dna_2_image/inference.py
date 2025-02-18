import argparse
import os
import re
import torch
import json
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for DNA sequence tasks.")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to save generated tokens.")
    return parser.parse_args()

def load_model(model_path='zehui127/Omni-DNA-DNA2Image'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to('cuda')
    return model, tokenizer

def generate(message, model, tokenizer, sample_num=49):
    tokenized_message = tokenizer(
        [message], return_tensors='pt', return_token_type_ids=False, add_special_tokens=True
    ).to('cuda')
    response = model.generate(**tokenized_message, max_new_tokens=sample_num, do_sample=False)
    reply = tokenizer.batch_decode(response, skip_special_tokens=False)[0]
    return extract_tokens(reply)

def extract_tokens(reply):
    numbers = re.findall(r'\b\d+\b', reply)
    numbers.insert(0, '4')
    while len(numbers) < 49:
        numbers.insert(0, '4')
    return " ".join(numbers)

def formatting_prompts_func(example):
    return {
        'formatted_text': [f"{seq}[MASK]" for seq in example['sequence']],
        'label': example['output'],
        'task_type': example['task']
    }

def group_by_task_type(dataset):
    task_types = set(dataset['task_type'])
    task_datasets = DatasetDict()
    for task_type in task_types:
        filtered_dataset = dataset.filter(lambda x: x['task_type'] == task_type, num_proc=1)
        if len(filtered_dataset) > 0:
            task_datasets[task_type] = filtered_dataset
            print(f"\nTask type '{task_type}': {len(filtered_dataset)} examples")
    return task_datasets

def inference(dataset, model, tokenizer,sample_limit_per_class=None):
    predictions, labels = [], []
    count = 0
    for element in tqdm(dataset):
        prediction = generate(element['formatted_text'], model, tokenizer)
        if prediction is not None:
            predictions.append(prediction)
            labels.append(element['label'])
        count += 1
        if sample_limit_per_class is not None:
            if count >= sample_limit_per_class:
                return predictions, labels
    return predictions, labels

def main():
    args = parse_args()
    model, tokenizer = load_model()
    os.makedirs(args.target_dir, exist_ok=True)

    raw_dataset = load_dataset("zehui127/Omni-DNA-Dataset-DNA2Image")
    dataset = raw_dataset['test'].map(
        formatting_prompts_func, batched=True, remove_columns=raw_dataset['test'].column_names,
        desc="Formatting dataset"
    )
    task_specific_datasets = group_by_task_type(dataset)
    sample_limit_per_class = 10
    tasks = ['TATAAA', 'CAAT', 'GGGCGG', 'TTAGGG']
    for task in tasks:
        if task in task_specific_datasets:
            dataset_test = task_specific_datasets[task]
            predictions, labels = inference(dataset_test, model, tokenizer,sample_limit_per_class)
            dir_name = str(labels[0]) if labels else "default"
            task_dir = os.path.join(args.target_dir, dir_name)
            os.makedirs(task_dir, exist_ok=True)

            for indx, pred in enumerate(predictions):
                filename = os.path.join(task_dir, f"{indx}.txt")
                with open(filename, "w") as file:
                    file.write(pred)

if __name__ == "__main__":
    main()
