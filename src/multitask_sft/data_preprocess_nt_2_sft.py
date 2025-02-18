import json
import os
from datasets import load_dataset
import argparse

def return_nt_dataset(task: str):
    """
    Fetch the train and test datasets for a specific nucleotide transformer task.
    """
    assert task in ['promoter_all', 'promoter_tata', 'promoter_no_tata',
                    'enhancers', 'enhancers_types', 'splice_sites_all', 'splice_sites_acceptors',
                    'splice_sites_donors', 'H3', 'H4', 'H3K9ac', 'H3K14ac', 'H4ac',
                    'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K36me3', 'H3K79me3'], "task not supported"

    dataset_train = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        name=task,
        split='train'
    )
    dataset_test = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        name=task,
        split='test'
    )
    return dataset_train, dataset_test

def save_jsonl(dataset, output_file):
    """
    Save a dataset to a JSONL file in the supervised fine-tuning format.
    """
    with open(output_file, "w") as f:
        for example in dataset:
            json_line = {
                "instruction": example["sequence"],
                "output": f"{str(example['label'])}",
                "task": example["task"]
            }
            f.write(json.dumps(json_line) + "\n")
    print(f"Dataset saved to {output_file}")

def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tasks = ['promoter_tata']

    combined_train = []
    combined_test = []

    for task in tasks:
        train_dataset, test_dataset = return_nt_dataset(task)
        combined_train.extend([{"task": task, "sequence": example["sequence"].upper(), "label": f"{''.join([str(example['label'])]*10)}"} for example in train_dataset])
        combined_test.extend([{"task": task, "sequence": example["sequence"].upper(), "label": f"{''.join([str(example['label'])]*10)}"} for example in test_dataset])

    train_output_file = os.path.join(output_dir, "train.jsonl")
    test_output_file = os.path.join(output_dir, "test.jsonl")

    save_jsonl(combined_train, train_output_file)
    save_jsonl(combined_test, test_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSONL files")
    args = parser.parse_args()
    main(args.output_dir)
