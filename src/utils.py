import pandas as pd
import random
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_annotation_corpus(raw_dataset):
    """
    A generator function that yields randomized samples from the 'clean_annotation' field
    of the provided dataset.
    Args:
        raw_dataset (DatasetDict): A dataset containing 'train' and 'test' splits.

    Yields:
        str: Randomized samples from the 'clean_annotation' field.
    """
    clean_annotations = []
    # Extract 'clean_annotation' from train and test splits
    for split in ['train', 'test']:
        if split in raw_dataset:
            clean_annotations.extend([record['output'] for record in raw_dataset[split] if 'output' in record and record['output']])
    # Shuffle the annotations
    random.shuffle(clean_annotations)
    # Yield each annotation
    for annotation in clean_annotations:
        yield annotation

def compute_added_vocabs(dataset='zehui127/Omni-DNA-Dataset-DNA2Text', ori_model="zehui127/Omni-DNA-1B",new_vocab_size=10000):
    raw_dataset = load_dataset(dataset)
    # Get annotation corpus as a generator
    annotation_corpus = get_annotation_corpus(raw_dataset)
    # Iterating through the generator to print examples
    for sample in annotation_corpus:
        print(sample)
        # Break early for demonstration
        break
    from transformers import AutoTokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(ori_model, trust_remote_code=True)
    # Step 1: Train a new tokenizer with a specified vocabulary size
    annot_tokenizer = old_tokenizer.train_new_from_iterator(annotation_corpus, new_vocab_size)
    # Step 2: Extract the new vocabulary
    new_vocab = list(annot_tokenizer.vocab.keys())
    # Step 3: Exclude tokens already present in the old tokenizer's vocabulary
    original_vocab = set(old_tokenizer.get_vocab().keys())
    filtered_new_vocab = [token for token in new_vocab if token not in original_vocab]
    return filtered_new_vocab

def extend_model_tokenizer(new_vocabs, output_path,ori_model="zehui127/Omni-DNA-1B"):
    old_tokenizer = AutoTokenizer.from_pretrained(ori_model, trust_remote_code=True)
    print(len(old_tokenizer))
    old_tokenizer.add_tokens(new_vocabs)
    print(len(old_tokenizer))  # Vocabulary size after adding the new tokens
    old_tokenizer.save_pretrained(output_path)
    model = AutoModelForCausalLM.from_pretrained(ori_model,trust_remote_code=True)
    model.resize_token_embeddings(len(old_tokenizer))
    model.save_pretrained(output_path, safe_serialization=False)
