import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import yaml
import torch.nn.functional as F  # Import torch.nn.functional for padding

import gdown
import zipfile
import os

from tqdm import tqdm
from datasets import load_dataset

def download_and_unzip_from_google_drive(output_dir='root/data'):
    file_id = "1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2"
    # Construct the Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the downloaded zip file and check if the data is already unzipped
    zip_path = os.path.join(output_dir, "downloaded_file.zip")

    # Check if the directory contains already unzipped files
    if any(os.scandir(output_dir)):
        print("Data already exists. Skipping download.")
        return

    # Download the file to the specified path
    gdown.download(url, zip_path, quiet=False)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Optionally, remove the zip file after extraction
    os.remove(zip_path)


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer=None):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        if tokenizer is not None:
            print("Processing the dataset...")
            res = []
            for text in tqdm(texts):
                encoded_text = tokenizer.encode(text)
                res.append(encoded_text)
            texts = res
        self.input_ids = texts
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def pad_and_insert_special_token(seqs, s_seq, e_seq, padding_token, target_length):
    padded_seqs = []
    for seq in seqs:
        # Calculate the padding needed
        pad_length = target_length - (len(seq) + 2)  # +2 for start and end tokens
        if pad_length > 0:
            # Pad the sequence with the padding token to the target length
            new_seq = [s_seq] + seq + [padding_token] * pad_length + [e_seq]
        elif pad_length == 0:
            # No padding needed, just add start and end tokens
            new_seq = [s_seq] + seq + [e_seq]
        else:
            # If the sequence is longer than target_length, truncate it and add start/end tokens
            new_seq = [s_seq] + seq[:target_length - 2] + [e_seq]
        # Append the padded sequence to the result list
        padded_seqs.append(new_seq)
    return torch.Tensor(padded_seqs).long()

def pad_and_insert_special_token_two_stage(seqs, s_seq, e_seq,
                                           padding_token,
                                           s_annot, e_annot, eos_token,
                                           seq_target_length=2000, annot_target_length=48):
    padded_seqs = []
    for seq in seqs:
        # Calculate the padding needed
        pad_length = seq_target_length - (len(seq) + 2)  # +2 for start and end tokens
        if pad_length > 0:
            # Pad the sequence with the padding token to the target length
            new_seq = [s_seq] + seq + [padding_token] * pad_length + [e_seq]
        elif pad_length == 0:
            # No padding needed, just add start and end tokens
            new_seq = [s_seq] + seq + [e_seq]
        else:
            # If the sequence is longer than target_length, truncate it and add start/end tokens
            new_seq = [s_seq] + seq[:seq_target_length - 2] + [e_seq]
        # Append the padded sequence to the result list
        # calculate annot padding
        annot_pad_length = annot_target_length - 2
        new_seq = new_seq + [s_annot] + [padding_token] * annot_pad_length + [eos_token]
        padded_seqs.append(new_seq)
    return torch.Tensor(padded_seqs).long()

def minimum_padding(input_ids, max_seq_len, padding_token=8193, s_seq=8195, e_seq=8196):
    padded_seqs = []
    for seq in input_ids:
        # Insert start sequence token (s_seq)
        seq = [s_seq] + seq
        # Convert to tensor before padding
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        # Calculate how many padding tokens are needed
        pad_length = max_seq_len - len(seq_tensor) - 1  # Subtract 1 for e_seq
        if pad_length > 0:
            # Pad between the sequence and e_seq
            seq_tensor = F.pad(seq_tensor, (0, pad_length), value=padding_token)
        # Add end sequence token (e_seq) after padding
        seq_tensor = torch.cat((seq_tensor, torch.tensor([e_seq], dtype=torch.long)))
        # Append the padded sequence
        padded_seqs.append(seq_tensor)
    # Stack the list of padded sequences into a tensor
    padded_seqs_tensor = torch.stack(padded_seqs)
    return padded_seqs_tensor


def cnn_collate_fn(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    # Extract input_ids and labels from instances
    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

    # Find the longest sequence in the batch
    max_len = max([x.size(0) for x in input_ids])  # max seq_len

    # Pad all sequences to the max length
    batch_padded = [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in input_ids]

    # Convert padded sequences to a tensor of dtype float32
    input_ids = torch.stack(batch_padded)  # Stack to create a batch tensor

    # Convert labels to tensor of type long
    labels = torch.tensor(labels, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # shape: (batch_size, max_seq_len, 4)
        labels=labels,        # shape: (batch_size)
    )

@dataclass
class PeftDataCollator(object):
    """Collate examples for supervised fine-tuning."""
    max_length: int
    pad_token_id: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)

        # Extract input_ids and labels
        model_inputs = [instance['input_ids'] for instance in instances]
        label_inputs = [instance['labels'] for instance in instances]
        attention_mask = [None] * batch_size
        # Process each sample in the batch
        for i in range(batch_size):
            sample_input_ids = model_inputs[i]
            label_input_ids =  [256] + [label_inputs[i]] + [self.pad_token_id]

            # Concatenate input and label sequences
            model_inputs[i] = sample_input_ids + label_input_ids
            label_inputs[i] = [-100] * len(sample_input_ids) + label_input_ids
            attention_mask[i] = [1] * len(model_inputs[i])

        # Padding the input_ids, attention_mask, and labels to max_length
        for i in range(batch_size):
            sample_input_ids = model_inputs[i]
            label_input_ids = label_inputs[i]

            # Padding input_ids and attention_mask
            model_inputs[i] = [self.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
            attention_mask[i] = [0] * (self.max_length - len(sample_input_ids)) + attention_mask[i]

            # Padding labels
            label_inputs[i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids

            # Convert to tensors and truncate to max_length
            model_inputs[i] = torch.tensor(model_inputs[i][:self.max_length], dtype=torch.long)
            attention_mask[i] = torch.tensor(attention_mask[i][:self.max_length], dtype=torch.long)
            label_inputs[i] = torch.tensor(label_inputs[i][:self.max_length], dtype=torch.long)

        # Return the processed inputs
        return dict(
            input_ids=torch.stack(model_inputs),
            attention_mask=torch.stack(attention_mask),
            labels=torch.stack(label_inputs),
        )

@dataclass
class CLSPeftDataCollator(object):
    """Collate examples for supervised fine-tuning."""
    max_length: int
    pad_token_id: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)

        # Extract input_ids and labels
        model_inputs = [instance['input_ids'] for instance in instances]
        label_inputs = [instance['labels'] for instance in instances]
        attention_mask = [None] * batch_size
        # Process each sample in the batch
        for i in range(batch_size):
            attention_mask[i] = [1] * len(model_inputs[i])

        # Padding the input_ids, attention_mask, and labels to max_length
        for i in range(batch_size):
            sample_input_ids = model_inputs[i]

            # Padding input_ids and attention_mask
            model_inputs[i] = [self.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
            attention_mask[i] = [0] * (self.max_length - len(sample_input_ids)) + attention_mask[i]

            # Convert to tensors and truncate to max_length
            model_inputs[i] = torch.tensor(model_inputs[i][:self.max_length], dtype=torch.long)
            attention_mask[i] = torch.tensor(attention_mask[i][:self.max_length], dtype=torch.long)

        # Return the processed inputs
        return dict(
            input_ids=torch.stack(model_inputs),
            attention_mask=torch.stack(attention_mask),
            labels=torch.tensor(label_inputs, dtype=torch.long),
        )

@dataclass
class CLSPeftMultiResDataCollator(object):
    """Collate examples for supervised fine-tuning."""
    max_length: int
    pad_token_id: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(instances)
        sample = instances[0]
        # Extract input_ids and labels
        model_inputs = [instance['input_ids'][0] for instance in instances]
        model_inputs_onehot = [instance['input_ids'][1] for instance in instances]
        label_inputs = [instance['labels'] for instance in instances]
        attention_mask = [None] * batch_size

        # process onehot
        # Find the longest sequence in the batch
        max_len_onehot = max([x.size(0) for x in model_inputs_onehot])  # max seq_len
        # Pad all sequences to the max length
        batch_padded = [F.pad(x, (0, 0, 0, max_len_onehot - x.size(0))) for x in model_inputs_onehot]

        # Process each sample in the batch
        for i in range(batch_size):
            attention_mask[i] = [1] * len(model_inputs[i])

        # Padding the input_ids, attention_mask, and labels to max_length
        for i in range(batch_size):
            sample_input_ids = model_inputs[i]

            # Padding input_ids and attention_mask
            model_inputs[i] = [self.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
            attention_mask[i] = [0] * (self.max_length - len(sample_input_ids)) + attention_mask[i]

            # Convert to tensors and truncate to max_length
            model_inputs[i] = torch.tensor(model_inputs[i][:self.max_length], dtype=torch.long)
            attention_mask[i] = torch.tensor(attention_mask[i][:self.max_length], dtype=torch.long)

        # Return the processed inputs
        return dict(
            input_ids=torch.stack(model_inputs),
            attention_mask=torch.stack(attention_mask),
            labels=torch.tensor(label_inputs, dtype=torch.long),
            onehot=torch.stack(batch_padded),
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    max_seq_len: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # TODO: replace 'N' with the padding token ID from a tokenizer
        input_ids =  minimum_padding(input_ids,
                                     max_seq_len=self.max_seq_len,
                                     padding_token=8193,
                                     s_seq=8195,
                                     e_seq=8196)
        # input_ids = pad_and_insert_special_token(input_ids,
        #                                          s_seq=8195,
        #                                          e_seq=8196,
        #                                          padding_token=8193,
        #                                          target_length=2048)
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def load_data_paths_from_yaml(config_path: str) -> Dict[str, str]:
    with open(config_path, 'r') as file:
        # Load the YAML file
        config = yaml.safe_load(file)
        # Extract the 'data_paths' dictionary
        data_paths = config.get('data_paths', {})
        root_folder = config.get('root_folder', None)
        if root_folder:
            # Update the data paths with the root folder
            data_paths = {key: os.path.join(root_folder, value) for key, value in data_paths.items()}
    return data_paths

def return_dataset(data_path: str, tokenizer) -> Tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    # call download_and_unzip_from_google_drive if the path does not exist
    if not os.path.exists(data_path):
        download_and_unzip_from_google_drive()
    train_dataset = SupervisedDataset(data_path=os.path.join(data_path, "train.csv"),
                                      tokenizer=tokenizer)
    val_dataset = SupervisedDataset(data_path=os.path.join(data_path, "dev.csv"),
                                    tokenizer=tokenizer)
    test_dataset = SupervisedDataset(data_path=os.path.join(data_path, "test.csv"),
                                     tokenizer=tokenizer)
    num_classes = max(train_dataset.num_labels,val_dataset.num_labels,test_dataset.num_labels)
    max_seq_len = max([max(len(instance['input_ids']) for instance in dataset) for dataset in [train_dataset, val_dataset, test_dataset]])
    # plus 2 for adding start_of_seq and end_of_seq token
    return train_dataset, val_dataset, test_dataset, num_classes, max_seq_len+2

class SupervisedDatasetMultiResolution(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer=None,
                 onehot_tokenizer=None):

        super(SupervisedDatasetMultiResolution, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        if tokenizer is not None:
            print("Processing the dataset...")
            res = []
            onehot_res = []
            for text in tqdm(texts):
                encoded_text = tokenizer.encode(text)
                onehot_text = onehot_tokenizer.encode(text)
                res.append(encoded_text)
                onehot_res.append(onehot_text)
            texts = res
        self.input_ids = texts
        self.onehot_ids = onehot_res
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=(self.input_ids[i],self.onehot_ids[i]), labels=self.labels[i])

def return_dataset_peft_mixed(data_path: str, tokenizer, onehot_tokenizer) -> Tuple[SupervisedDatasetMultiResolution, SupervisedDatasetMultiResolution, SupervisedDatasetMultiResolution]:
    # call download_and_unzip_from_google_drive if the path does not exist
    if not os.path.exists(data_path):
        download_and_unzip_from_google_drive()
    train_dataset = SupervisedDatasetMultiResolution(data_path=os.path.join(data_path, "train.csv"),
                                      tokenizer=tokenizer,
                                      onehot_tokenizer=onehot_tokenizer)
    val_dataset = SupervisedDatasetMultiResolution(data_path=os.path.join(data_path, "dev.csv"),
                                                   tokenizer=tokenizer,
                                                   onehot_tokenizer=onehot_tokenizer)
    test_dataset = SupervisedDatasetMultiResolution(data_path=os.path.join(data_path, "test.csv"),
                                                    tokenizer=tokenizer,
                                                    onehot_tokenizer=onehot_tokenizer)
    num_classes = max(train_dataset.num_labels,val_dataset.num_labels,test_dataset.num_labels)
    max_seq_len = max([max(len(instance['input_ids'][0]) for instance in dataset) for dataset in [train_dataset, val_dataset, test_dataset]])
    # plus 2 for adding start_of_seq and end_of_seq token
    return train_dataset, val_dataset, test_dataset, num_classes, max_seq_len
#######################
# Example usage
#######################


# download_and_unzip_from_google_drive("root/data")
# data_paths = load_data_paths_from_yaml('config.yaml')
# train_dataset, val_dataset, test_dataset = return_dataset(data_paths['prom_300_tata'])
# test_iter = iter(test_dataset)
# test_batch = next(test_iter)
# print(test_batch)

class SupervisedDatasetNew(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer=None):

        super(SupervisedDatasetNew, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        if tokenizer is not None:
            print("Processing the dataset...")
            res = []
            for text in tqdm(texts):
                encoded_text = tokenizer(text)['input_ids']
                res.append(encoded_text)
            texts = res
        self.input_ids = texts
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def return_dataset_new(data_path: str, tokenizer) -> Tuple[SupervisedDataset, SupervisedDataset, SupervisedDataset]:
    # call download_and_unzip_from_google_drive if the path does not exist
    if not os.path.exists(data_path):
        download_and_unzip_from_google_drive()
    test_dataset = SupervisedDatasetOri(data_path=os.path.join(data_path, "test.csv"),
                                     tokenizer=tokenizer)
    train_dataset = SupervisedDatasetOri(data_path=os.path.join(data_path, "train.csv"),
                                      tokenizer=tokenizer)
    val_dataset = SupervisedDatasetOri(data_path=os.path.join(data_path, "dev.csv"),
                                       tokenizer=tokenizer)

    num_classes = max(train_dataset.num_labels,val_dataset.num_labels,test_dataset.num_labels)
    max_seq_len = max([max(len(instance['input_ids']) for instance in dataset) for dataset in [train_dataset, val_dataset, test_dataset]])
    # plus 2 for adding start_of_seq and end_of_seq token
    return train_dataset, val_dataset, test_dataset, num_classes, max_seq_len+2

class SupervisedDatasetOri(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDatasetOri, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            # remove characters that are not ACGT within texts
            # texts = [''.join([c for c in text if c in 'ACGT']) for text in texts]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        print(f"sample attention mask: {self.attention_mask[0]}")
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorLastest(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

@dataclass
class DataCollatorLastestCaduceus(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

def return_genomic_bench_dataset(data_path: str, tokenizer, seed=42):
    genomics = {
                'demo_coding_vs_intergenomic_seqs': 'DemoCodingVsIntergenomicSeqs',
                'demo_human_or_worm': 'DemoHumanOrWorm',
                'drosophila_enhancers_stark': 'DrosophilaEnhancersStark',
                'dummy_mouse_enhancers_ensembl': 'DemoMouseEnhancers',
                'human_enhancers_cohn': 'HumanEnhancersCohn',
                'human_enhancers_ensembl': 'HumanEnhancersEnsembl',
                'human_nontata_promoters': 'HumanNontataPromoters',
                'human_ocr_ensembl': 'HumanOcrEnsembl'}
    if data_path == "human_ensembl_regulatory":
        from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
        train_dset = get_dataset(data_path, 'train')
        test_dset = get_dataset(data_path,'test')
    else:
        data_set = genomics[data_path]
        # from genomic_benchmarks.dataset_getters.pytorch_datasets import data_set
        import importlib
        module = importlib.import_module("genomic_benchmarks.dataset_getters.pytorch_datasets")
        data_set = getattr(module, data_set)

        train_dset = data_set('train', version=0)
        test_dset = data_set('test', version=0)
        print(f"succesfully loaded {data_path} dataset")
        print(f"original max seq len {max([len(seq) for seq,label in train_dset])}")

    dataset_train, val_dataset = split_train_val(train_dset, val_split=0.1,seed=seed)
    train_dataset = SupervisedDatasetGB(data=dataset_train,
                                      tokenizer=tokenizer)
    val_dataset = SupervisedDatasetGB(data=val_dataset,
                                      tokenizer=tokenizer)
    test_dataset = SupervisedDatasetGB(data=test_dset,
                                    tokenizer=tokenizer)
    num_classes = max(train_dataset.num_labels,test_dataset.num_labels)
    max_seq_len = max([max(len(instance['input_ids']) for instance in dataset) for dataset in [train_dataset, val_dataset,test_dataset]])
    return train_dataset, val_dataset, test_dataset, num_classes, max_seq_len+2


class SupervisedDatasetGB(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDatasetGB, self).__init__()
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
            # iterate through the text, label pair, remove the all N text and corresponding labels
            # and remove non ACGT characters
            ######### uncomments below to remove all non N############
            filtered_texts = []
            filtered_labels = []
            for text, label in zip(texts, labels):
                # Remove sequences that are all 'N'
                if text.count('N') == len(text):
                    continue
                # Remove non-ACGT characters
                cleaned_text = ''.join(c for c in text if c in 'ACGT')
                if cleaned_text:  # Only add if there's text remaining
                    filtered_texts.append(cleaned_text)
                    filtered_labels.append(label)
            texts = filtered_texts
            labels = filtered_labels
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # print the average and max length of the data
        max_len = 0
        avg_len_arr = []
        max_seq = None
        for text in texts:
            seq_len = len(tokenizer.tokenize(text))
            if max_len < seq_len:
                max_seq = text
                max_len = seq_len
            avg_len_arr.append(seq_len)
        print(f"max_seq_len is {max_len}")
        print(f"max_seq is {max_seq}")
        print(f"avg_seq_len is {sum(avg_len_arr)/len(avg_len_arr)}")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        print(f"sample attention mask: {self.attention_mask[0]}")
        self.labels = labels
        self.num_labels = len(set(labels))
        print(f"num labels is {set(labels)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def return_nt_dataset(task: str, tokenizer,seed=42):
    assert task in ['promoter_all', 'promoter_tata', 'promoter_no_tata',
                    'enhancers', 'enhancers_types', 'splice_sites_all', 'splice_sites_acceptors',
                    'splice_sites_donors', 'H3', 'H4', 'H3K9ac', 'H3K14ac', 'H4ac',
                    'H3K4me1','H3K4me2','H3K4me3','H3K36me3','H3K79me3'], "task not supported"
    dataset_train = load_dataset(
                    "InstaDeepAI/nucleotide_transformer_downstream_tasks",
                    name=task,
                    split='train',
                    trust_remote_code=True
                )
    dataset_train, dataset_val = split_train_val(dataset_train, val_split=0.05,seed=seed)
    dataset_test = load_dataset(
                    "InstaDeepAI/nucleotide_transformer_downstream_tasks",
                    name=task,
                    split='test',
                    trust_remote_code=True
                )
    train_dataset = SupervisedDatasetNT(data_path=dataset_train, tokenizer=tokenizer)
    val_dataset = SupervisedDatasetNT(data_path=dataset_val, tokenizer=tokenizer)
    test_dataset = SupervisedDatasetNT(data_path=dataset_test, tokenizer=tokenizer)
    num_classes = max(train_dataset.num_labels,val_dataset.num_labels,test_dataset.num_labels)
    max_seq_len = max([max(len(instance['input_ids']) for instance in dataset) for dataset in [train_dataset, val_dataset, test_dataset]])
    # plus 2 for adding start_of_seq and end_of_seq token
    return train_dataset, val_dataset, test_dataset, num_classes, max_seq_len+2

def split_train_val(dataset_train, val_split=0.1,seed=42):
    """
    Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
    """
    train_len = int(len(dataset_train) * (1.0 - val_split))
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train,
        (train_len, len(dataset_train) - train_len),
        generator=torch.Generator().manual_seed(
            seed
        ),
    )
    return dataset_train, dataset_val



class SupervisedDatasetNT(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

        super(SupervisedDatasetNT, self).__init__()

        data = data_path
        if len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform NT benchmark classification...")
            texts = [d['sequence'] for d in data]
            labels = [int(d['label']) for d in data]
            # remove characters that are not ACGT within texts
            # texts = [''.join([c for c in text if c in 'ACGT']) for text in texts]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        if "attention_mask" in output:
            self.attention_mask = output["attention_mask"]
            print(f"sample attention mask: {self.attention_mask[0]}")
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
