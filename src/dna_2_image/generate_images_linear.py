import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from .vq_vae_best_mint_updated import VQVAE, training_data_filter, validation_data_filter
from torchvision.utils import save_image
from tqdm import tqdm
import argparse


config = {
        'batch_size': 128,
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 6,
        'commitment_cost': 0.25,
        'decay': 0.99,
        'learning_rate': 5e-4,
        'max_epochs': 100,
    }

class SaveMNISTTokens:
    def __init__(self, model, data_loader, output_dir, device, unique_tokens={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5'}):
        self.model = model
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.device = device
        self.unique_tokens = unique_tokens

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Create subdirectories for each class
        for i in range(4):
            class_dir = os.path.join(self.output_dir, str(i))
            os.makedirs(class_dir, exist_ok=True)

    def map_images_to_tokens(self):
        self.model.eval()
        token_list = list(self.unique_tokens.keys())

        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(enumerate(self.data_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Pass images through the model to get quantized indices
                z = self.model.model._encoder(images)
                z = self.model.model._pre_vq_conv(z)
                _, _, _, encodings = self.model.model._vq_vae(z)

                # Convert encodings to linearized tokens
                one_hot_array = encodings.cpu().numpy()
                tokens = indices_to_token(one_hot_array, token_list)

                # Save tokens to corresponding class folders
                class_folder = os.path.join(self.output_dir, str(labels.item()))
                token_path = os.path.join(class_folder, f"{batch_idx}.txt")
                with open(token_path, "w") as token_file:
                    token_file.write(" ".join(map(str, tokens)))

    def generate_images_from_tokens(self, tokens_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        token_list = list(self.unique_tokens.keys())

        for class_name in os.listdir(tokens_dir):
            class_folder = os.path.join(tokens_dir, class_name)
            output_class_folder = os.path.join(output_dir, class_name)
            os.makedirs(output_class_folder, exist_ok=True)

            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)

                # Load tokens and convert to one-hot encoded indices
                with open(file_path, "r") as token_file:
                    tokens = list(map(str, token_file.read().strip().split()))
                one_hot_array = token_to_indices(tokens, token_list)
                one_hot_tensor = torch.tensor(one_hot_array, device=self.device)
                if one_hot_tensor.shape[0] != 49:
                    print(f"warning: invalid shape {one_hot_tensor.shape}, {file_path}")
                    continue
                # Decode from quantized representation
                quantized = torch.matmul(
                    one_hot_tensor, self.model.model._vq_vae._embedding.weight
                ).view(1, 7, 7, config['embedding_dim'])

                quantized = quantized.permute(0, 3, 1, 2).contiguous()
                images = self.model.model._decoder(quantized)

                # Save the generated image
                save_path = os.path.join(output_class_folder, file_name.replace(".txt", ".png"))
                save_image(images.squeeze(0), save_path)


# Helper functions for token conversion
def indices_to_token(one_hot_array, token_list):
    """
    Convert a one-hot encoded numpy array to a list of linearized tokens.

    Args:
        one_hot_array (np.ndarray): A numpy array with one-hot encodings.
        token_list (list): List of unique tokens.

    Returns:
        list: A list of tokens.
    """
    linearized_tokens = []
    for row in one_hot_array:
        index = np.argmax(row)
        linearized_tokens.append(token_list[index])
    return linearized_tokens


def token_to_indices(tokens, token_list):
    """
    Convert a list of tokens to a one-hot encoded numpy array.

    Args:
        tokens (list): A list of tokens.
        token_list (list): List of unique tokens.

    Returns:
        np.ndarray: One-hot encoded numpy array.
    """
    # print(tokens)
    token_to_index = {str(token): idx for idx, token in enumerate(token_list)}
    # print(token_to_index)
    valid_keys = token_to_index.keys()
    one_hot_array = np.zeros((len(tokens), len(token_list)), dtype=np.float32)
    for i, token in enumerate(tokens):
        if token not in valid_keys:
            token = '1'
        index = token_to_index[token]
        one_hot_array[i, index] = 1
    return one_hot_array



def main():
    parser = argparse.ArgumentParser(description="Run VQVAE index saving and image reconstruction.")
    parser.add_argument("--output_indices", type=str, required=True, help="Directory to save output indices.")
    parser.add_argument("--reconstructed_images_dir", type=str, required=True, help="Directory to save reconstructed images.")
    args = parser.parse_args()


    # Load the checkpoint
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_file_path, "vqvae-epoch=90-val_loss=0.00.ckpt")
    loaded_model = VQVAE.load_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.eval()
    loaded_model.to(device)

    # Dataloader
    combined_dataset = torch.utils.data.ConcatDataset([training_data_filter, validation_data_filter])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=1,  # Keeping batch size 1 for SaveMNISTIndices implementation
        shuffle=True,
        num_workers=4
    )

    # Output directory for indices
    output_dir = args.output_indices

    # Create and run the saver
    index_saver = SaveMNISTTokens(loaded_model, combined_loader, output_dir, device)
    # index_saver.map_images_to_tokens()

    # Generate images from saved indices
    reconstructed_images_dir = args.reconstructed_images_dir
    index_saver.generate_images_from_tokens(output_dir, reconstructed_images_dir)

    print(f"Indices saved to '{output_dir}' folder.")
    print(f"Reconstructed images saved to '{reconstructed_images_dir}' folder.")


if __name__ == "__main__":
    main()
