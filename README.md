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
   pip install datasets ai2-olmo trl==0.13 transformers datasets
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

- **Regulatory Element Classification:** Enhancer/promoter/splice site detection
- **Histone Modification Prediction:** Acetylation and methylation state identification
- **Genomic Function Annotation:** DNA-to-text mapping (DNA2Function)
- **Cross-modal Learning:** DNA-to-image mapping (DNA2Image)
- **Multi-task Learning:** A single model can solve multiple tasks simultaneously

---

## Usage

### As a Generative AutoRegressive Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model
model_tokenizer_path = "zehui127/Omni-DNA-1B"
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path).to('cuda')

def generate(message, task_type, model=model, sample_num=1):
    tokenized_message = tokenizer(
        [message], return_tensors='pt', return_token_type_ids=False, add_special_tokens=True
    ).to('cuda')
    response = model.generate(**tokenized_message, max_new_tokens=sample_num, do_sample=False)
    reply = tokenizer.batch_decode(response, skip_special_tokens=False)[0]
    return reply.replace(" ", "")

# Example usage:
message = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTA"
output = generate(message, "DNA sequence classification")
print(f"Generated output: {output}")
```

### Attaching Classification Head

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("zehui127/Omni-DNA-1B", num_labels=2, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zehui127/Omni-DNA-1B", trust_remote_code=True)

sequence = "ATGCGTACGTAGCTAGCTAGCTAGCTAGCTA"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(dim=-1).item()
print(f"Predicted class: {predicted_class}")
```

---

## Supervised Finetuning (SFT) Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

model_name = "zehui127/Omni-DNA-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
