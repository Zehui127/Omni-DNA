from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets,concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import os
import torch
import sys
from transformers import AutoModelForCausalLM, TrainingArguments, AutoModelForSequenceClassification
import transformers
import argparse
from ..utils import compute_added_vocabs, extend_model_tokenizer

def group_by_task_type(dataset):
    # First, get unique task types
    task_types = set(dataset['task'])
    # Create datasets for each task type using map
    task_datasets = DatasetDict()
    for task_type in task_types:
        # Define a filter function for this task type
        def filter_task(example, task=task_type):
            return example['task'] == task
        # Apply the filter using map
        filtered_dataset = dataset.filter(
            filter_task,
            num_proc=1,  # Adjust based on your CPU cores
            desc=f"Filtering {task_type} examples"
        )
        if len(filtered_dataset) > 0:
            task_datasets[task_type] = filtered_dataset
            print(f"\nTask type '{task_type}':")
            print(f"Number of examples: {len(filtered_dataset)}")
    return task_datasets


def run_sft(output_path):
    model_tokenizer_path = "zehui127/Omni-DNA-1B"
    tasks = ['enhancers', 'H3', 'H4', 'H3K9ac', 'H3K14ac', 'H4ac',
             'H3K4me1','H3K4me2','H3K4me3','H3K36me3','H3K79me3']
    # extend model emb matrixa and tokenizer
    added_vocabs = tasks
    extend_model_tokenizer(added_vocabs,output_path)
    model_tokenizer_path = output_path

    raw_dataset = load_dataset("zehui127/Omni-DNA-dataset-nt-downstream-multitask")
    dataset = raw_dataset['train']
    # group by task type
    task_specific_datasets = group_by_task_type(dataset)
    # get the tasks
    dataset_list = [task_specific_datasets[task] for task in tasks]
    dataset = concatenate_datasets(dataset_list)
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path,trust_remote_code=True)
    # define the formatting function for nt tasks
    def formatting_prompts_func(example):
        text = f"{example['instruction']}[MASK]{example['output']}"
        return text

    response_template = "[MASK]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # define args for training
    training_args = SFTConfig(
        per_device_train_batch_size=3, # 8 GPUs are used
        per_device_eval_batch_size=6,
        save_total_limit=1,
        max_seq_length=512,
        output_dir=f"{output_path}",
        save_safetensors=False,
        num_train_epochs=10,
        save_strategy="epoch",
        neftune_noise_alpha=5, # add NEFt
    )
    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    # Train the model
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Supervised finetuning for multi-tasking")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for saving the output models")
    args = parser.parse_args()
    run_sft(args.output_path)

if __name__ == "__main__":
    main()
