import argparse
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from ..utils import compute_added_vocabs, extend_model_tokenizer

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"{example['instruction'][i]}[MASK]{example['output'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    MAX_LEN = 580
    parser = argparse.ArgumentParser(description="Train a model using a formatted DNA dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    model_tokenizer_path = 'zehui127/Omni-DNA-1B'
    # first extend the tokenizer and embedding matrix of the model
    added_vocabs = compute_added_vocabs("zehui127/Omni-DNA-Dataset-DNA2Text")
    extend_model_tokenizer(added_vocabs,args.output_dir)
    model_tokenizer_path = args.output_dir

    # model_tokenizer_path = "/home/v-zehuili/mnt/data/section6_model/extended_model/8192_vocab"
    # start the sft
    raw_dataset = load_dataset("zehui127/Omni-DNA-Dataset-DNA2Text")
    dataset = raw_dataset['train']
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path)

    response_template = "[MASK]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = SFTConfig(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=1,
        max_seq_length=MAX_LEN,
        output_dir=f"{args.output_dir}/models",
        save_safetensors=False,
        num_train_epochs=10,
        neftune_noise_alpha=5, # add NEFt
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )

    trainer.train()

if __name__ == "__main__":
    main()
