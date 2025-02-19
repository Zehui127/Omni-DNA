# Omni-DNA
Omni-DNA is a cross-modal, multi-task genomic foundation model designed to generalize across diverse genomic tasks.

---
## News
- 2024.02.10: Release [Omni-DNA](https://huggingface.co/collections/zehui127/omni-dna-67a2230c352d4fd8f4d1a4bd) models on HuggingFace, and the [Preprint](https://arxiv.org/abs/2502.03499)

---

## Installation

1. Create a virtual environment:
   ```bash
   conda create -n omni_dna python=3.10 -y
   conda activate omni_dna
   ```
2. Install dependencies:
   ```bash
   pip install trl==0.13 transformers datasets datasets ai2-olmo
   # for replicating the dna2image, the following package are also needed
   # pip install torchvision matplotlib pytorch_lightning
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/Zehui127/Omni-DNA.git
   cd Omni-DNA
   ```
---

## Model Details

### Base Models

| Size          | Training Tokens | Layers | Hidden Size | Attention Heads | Context Length | Hugging Face Identifier |
|--------------|----------------|--------|-------------|-----------------|----------------|--------------------------|
| Omni-DNA 20M  | 300B           | 8      | 256         | 8               | 250            | [Omni-DNA-20M](https://huggingface.co/zehui127/Omni-DNA-20M)           |
| Omni-DNA 60M  | 300B           | 8      | 512         | 8               | 250            | [Omni-DNA-60M](https://huggingface.co/zehui127/Omni-DNA-60M)           |
| Omni-DNA 116M | 300B           | 12     | 768         | 16              | 250            | [Omni-DNA-116M](https://huggingface.co/zehui127/Omni-DNA-116M)          |
| Omni-DNA 300M | 300B           | 16     | 1024        | 16              | 250            | [Omni-DNA-300M](https://huggingface.co/zehui127/Omni-DNA-300M)          |
| Omni-DNA 700M | 300B           | 16     | 1536        | 16              | 250            | [Omni-DNA-700M](https://huggingface.co/zehui127/Omni-DNA-700M)          |
| Omni-DNA 1B   | 300B           | 16     | 2048        | 16              | 250            | [Omni-DNA-1B](https://huggingface.co/zehui127/Omni-DNA-1B)            |

### SFT Models
| Model Name               | Base Model |  Hugging Face Identifier |
|--------------------------|---------------|----------------|
| Omni-DNA-Multitask       | Omni-DNA 1B           | [Omni-DNA-Multitask](https://huggingface.co/zehui127/Omni-DNA-Multitask)            |
| Omni-DNA-DNA2Function    | Omni-DNA 1B           | [Omni-DNA-DNA2Function](https://huggingface.co/zehui127/Omni-DNA-DNA2Function)            |
| Omni-DNA-DNA2Image       | Omni-DNA 1B           | [Omni-DNA-DNA2Image](https://huggingface.co/zehui127/Omni-DNA-DNA2Image)            |

---

## Capabilities

Omni-DNA is trained to perform **multiple genomic tasks** including:

- **Finetuning Base Models with MLP attached:** This is the same as existing Genomic Foundation Models. See `src/FT_CLS_Head`, which shows classification on Genomic Benchmarks and Nucleotide Transformer Downstream tasks.
- **Supervised FineTuning (SFT) for Multitasking and Cross-Modality Generation:** We show Multi-tasking examples in `src/multitask_sft`, and dna2text examples in `src/dna_2_text`.
- **SFT for Customized Generation Task**: You could follow the same code as Multitasking and Cross-Modality Generation. But you need to prepare the dataset and then use `src/utils` to extend the vocab sizes of the base model. Examples comes later.
---

## Examples
## Finetuning Base Models with MLP attached
You need to define your own data loader below are examples of performing ft on gb and nt
```python
import os
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoModelForSequenceClassification, set_seed, AutoTokenizer, Trainer
from torch.utils.data import DataLoader
from src.datasets.dataloaders import DataCollatorLastest, return_nt_dataset, return_genomic_bench_dataset
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

```
## Supervised Finetuning (SFT) Example

Given an example Json File

```Json
[
  {
    "instruction": "ATGCGTAC",
    "task": "TASK1:complementary DNA strand",
    "output": "TACGCATG"
  },
  {
    "instruction": "CGCATAT",
    "task": "TASK1:complementary DNA strand",
    "output": "GCGTATA"
  },
  {
    "instruction": "GCGAGATATAAAAA",
    "task": "TASK2:Classify the given DNA sequence based on its function.",
    "output": "Class: Promoter region"
  }
]
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from src.utils import compute_added_vocabs, extend_model_tokenizer

# define newly added vocabs and a local path to save extended model
# see src/utils.py/compute_added_vocabs for how to compute the added vocabs
added_vocabs = ...
path_for_extended_model = ...
extend_model_tokenizer(added_vocabs,path_for_extended_model)
# load extended model
model = AutoModelForCausalLM.from_pretrained(path_for_extended_model)
tokenizer = AutoTokenizer.from_pretrained(path_for_extended_model)

dataset = load_dataset("json", data_files={"train": "path/to/train.json"})
dataset = dataset["train"]
def formatting_prompts_func(example):
    return [f"{example['instruction']} {example['task']} [SEP] {example['output']}"]

response_template = "[SEP]"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(
    per_device_train_batch_size=6,
    per_device_eval_batch_size=8,
    max_seq_length=512,
    output_dir="./finetuned_omni_dna",
    num_train_epochs=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```

## Replicating Experiments in the Paper
Run the bash scirpt `run.sh`
```bash
export PYTHONPATH=$(pwd)

# finetuning with MLP on Genomic Benchmark and Nucleotide Transformer Downstream tasks.
python scripts/cls_head_ft.py --dataset nt_downstream --task promoter_tata --model zehui127/Omni-DNA-116M --seed 123 --learning_rate 0.000005 --batch_size 8 --num_of_epoch 10
# or finetuning with MLP with hyperparamter sweeping
python scripts/cls_head_ft_sweep.py

# inference with multi-tasking model
python scripts/sft_multitask.py --model_tokenizer_path zehui127/Omni-DNA-Multitask

# inference with dna 2 text model, output_path to save output results
python scripts/dna_2_text.py --output_path current_working_dir

# inference with dna 2 image model, target_dir to save output results
python scripts/dna_2_image_indices.py --target_dir current_working_dir
## then generate the images from generated discrete tokens
python scripts/dna_2_image_images.py --output_indices current_working_dir --reconstructed_images_dir current_working_dir

```
---

## Citation

If you use Omni-DNA in your research, please cite:

```bibtex
@article{li2025omni,
  title={Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning},
  author={Li, Zehui and Subasri, Vallijah and Shen, Yifei and Li, Dongsheng and Zhao, Yiren and Stan, Guy-Bart and Shan, Caihua},
  journal={arXiv preprint arXiv:2502.03499},
  year={2025}
}
```

---

## License

Omni-DNA is released under the **MIT License**.

---

## Acknowledgements

- We highly appreciate all dataset providers for making genomic datasets publicly available.
- Thanks to Microsoft Research Asia for providing computational support.
- Special thanks to the developers of the OLMo model for inspiration and tools.

---

## Contact

For research inquiries, contact **Zehui Li** at `zl6222@ic.ac.uk` or raise Issues through github.

---
