import argparse
import os
import torch
import re
import numpy as np
import sklearn
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Run DNA task inference with a specified model and tokenizer.")
    parser.add_argument("--model_tokenizer_path", type=str, required=True, help="Path to the pretrained model and tokenizer.")
    return parser.parse_args()

def load_model_and_tokenizer(model_tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path).to('cuda')
    return model, tokenizer

def generate(message, task_type, model, tokenizer, sample_num=1):
    tokenized_message = tokenizer([message], return_tensors='pt', return_token_type_ids=False, add_special_tokens=True).to('cuda')
    response = model.generate(**tokenized_message, max_new_tokens=sample_num, do_sample=False)
    reply = tokenizer.batch_decode(response, skip_special_tokens=False)[0].replace(" ", "")
    return extract_label(reply, task_type)

def extract_label(message, task_type):
    task_type = '[MASK]'
    answer = message.split(task_type)[1]
    match = re.search(r'\d+', answer)
    return match.group() if match else None

def load_and_format_dataset():
    raw_dataset = load_dataset("zehui127/Omni-DNA-dataset-nt-downstream-multitask")
    dataset = raw_dataset['test']

    def formatting_prompts_func(example):
        output_texts = [f"{instr}[MASK]" for instr in example['instruction']]
        labels = [output[-1] for output in example['output']]
        task_types = example['task']
        return {'formatted_text': output_texts, 'label': labels, 'task_type': task_types}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names, desc="Formatting dataset")
    return formatted_dataset

def group_by_task_type(dataset):
    task_types = set(dataset['task_type'])
    task_datasets = DatasetDict()

    for task_type in task_types:
        filtered_dataset = dataset.filter(lambda x: x['task_type'] == task_type, num_proc=1, desc=f"Filtering {task_type} examples")
        if len(filtered_dataset) > 0:
            task_datasets[task_type] = filtered_dataset
            print(f"\nTask type '{task_type}': {len(filtered_dataset)} examples")

    return task_datasets

def calculate_metrics(predictions, labels):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

def inference(dataset, model, tokenizer):
    predictions, labels = [], []

    for element in tqdm(dataset):
        prediction = generate(element['formatted_text'], element['task_type'], model, tokenizer)
        sample_num = 2
        while prediction is None:
            prediction = generate(element['formatted_text'], element['task_type'], model, tokenizer, sample_num)
            sample_num += 1
            if sample_num >= 20:
                prediction = '0'
                print("Warning: No valid result")
                break
        predictions.append(int(str(prediction)[0]))
        labels.append(int(element['label']))

    return calculate_metrics(np.array(predictions), np.array(labels))

def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_tokenizer_path)
    formatted_dataset = load_and_format_dataset()
    task_specific_datasets = group_by_task_type(formatted_dataset)

    tasks = ['H3', 'H4', 'H3K9ac', 'H3K14ac', 'H4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K36me3', 'H3K79me3']

    for task in tasks:
        print(f"==========={task}=========")
        dataset_test = task_specific_datasets.get(task, None)
        if dataset_test:
            print(inference(dataset_test, model, tokenizer))
        else:
            print(f"No data for task {task}")

if __name__ == "__main__":
    main()
