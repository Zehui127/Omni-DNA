import argparse
from .omni_dna import run_finetune
import wandb
import yaml

def run(dataset_name, task, model_type):
    run = wandb.init()
    run_finetune(dataset=dataset_name,
                 task=task,
                 seed=wandb.config.seed,
                 model_type=model_type,
                 learning_rate=wandb.config.learning_rate,
                 batch_size=wandb.config.batch_size,
                 num_of_epoch=wandb.config.num_of_epoch)

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning with a specified dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use for training')
    parser.add_argument('--task', type=str, required=True, help='Name of the task')
    parser.add_argument('--model', type=str, required=True, help='Model type')

    args = parser.parse_args()

    with open("/home/v-zehuili/repositories/Omni-DNA/src/FT_CLS_Head/sweep_config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=config, project="SSF")
    wandb.agent(sweep_id, function=lambda: run(args.dataset, args.task, args.model))

if __name__ == "__main__":
    main()
