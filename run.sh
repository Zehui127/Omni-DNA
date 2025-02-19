#!/bin/bash
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
