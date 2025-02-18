import os
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoModelForSequenceClassification, set_seed, AutoTokenizer, Trainer
from torch.utils.data import DataLoader
from ..datasets.dataloaders import DataCollatorLastest, return_nt_dataset, return_genomic_bench_dataset
import numpy as np
import sklearn
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import csv
import argparse
import shutil

valid_omni_dna_path = {
    "zehui127/Omni-DNA-1B",
    "zehui127/Omni-DNA-20M",
    "zehui127/Omni-DNA-60M",
    "zehui127/Omni-DNA-116M",
    "zehui127/Omni-DNA-300M",
    "zehui127/Omni-DNA-700M",
}

dataset_loader = {
    "gb": return_genomic_bench_dataset,
    "nt_downstream": return_nt_dataset,
}

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gb, nt_downstream)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., promoter_tata)")
    parser.add_argument("--model", type=str, required=True, help="Model type (e.g., olmo, nt, dnabert2, hyenaDNA, caduceus)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed value for training")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per device")
    parser.add_argument("--num_of_epoch", type=int, required=True, help="Number of training epochs")

    args = parser.parse_args()
    print(f"###### Running fine-tune model on task '{args.task}' with seed '{args.seed}' using model '{args.model}'...")
    run_finetune(args.dataset, args.task, args.seed, args.model, args.learning_rate, args.batch_size, args.num_of_epoch)

def run_finetune(dataset, task, seed, model_type, learning_rate, batch_size, num_of_epoch, MAX_LEN=1000, path_prefix="saved_models"):
    assert model_type in valid_omni_dna_path, "Model not supported"
    assert dataset in ["gb", "nt_downstream"], "Dataset should be one of [gb, nt_downstream]"
    return_data_loader = dataset_loader[dataset]
    set_seed(seed)

    cache_dir = f"{path_prefix}/cache_directory"
    results_file = f"{path_prefix}/results_{model_type}.csv"
    # make dir for results_file if not exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    training_args = TrainingArguments(
        output_dir=f"{path_prefix}/output_model",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_of_epoch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        max_grad_norm=1.0,
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_safetensors=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    tokenizer.model_max_length = MAX_LEN
    train_data, val_data, test_data, class_num, max_seq_len = return_data_loader(task, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=class_num,trust_remote_code=True)
    collate_fn = DataCollatorLastest(tokenizer=tokenizer)
    print(f"!!!!!!MAX LEN IS {max_seq_len}")

    trainer = Trainer(
        model=model,
        tokenizer=None,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn
    )
    trainer.train(resume_from_checkpoint=False)

    print("\nTesting the model on the test dataset...\n")
    test_metrics = trainer.evaluate(eval_dataset=test_data)
    print(f"Test Metrics: {test_metrics}")
    write_header = not os.path.exists(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Task", "Seed", "Model Type", "Learning Rate", "Batch Size", "Epochs"] + list(test_metrics.keys()))
        writer.writerow([task, seed, model_type, learning_rate, batch_size, num_of_epoch] + list(test_metrics.values()))

    print(f"Test metrics appended to {results_file}")

if __name__ == "__main__":
    main()
