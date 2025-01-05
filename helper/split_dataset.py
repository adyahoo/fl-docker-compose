import torch
import os
import argparse
import numpy as np

from torchvision import datasets, transforms
from collections import defaultdict

parser = argparse.ArgumentParser(description="Flower Split Dataset")
parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
parser.add_argument("--total_samples", type=int, default=10000, help="Total number of samples")

args = parser.parse_args()

def split_mnist(dataset, num_clients, total_samples, output_dir):
    """
    Splits the MNIST dataset into parts with specified sizes, ensuring each part has all labels.

    Args:
        dataset: PyTorch dataset (MNIST in this case).
        client_sizes: List of integers specifying the number of samples for each client.
        output_dir: Directory where the splits will be saved.
    """
    # Create output directories for clients
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_clients):
        os.makedirs(os.path.join(output_dir, f"client_{i+1}"), exist_ok=True)

    # Generate random sample sized for each client
    sample_sizes = np.random.multinomial(total_samples, [1/num_clients] * num_clients)
    print(f"Sample sizes: {sample_sizes}")
    
    # Group dataset indices by label
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    # Shuffle indices for randomness
    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])

    # Assign data to clients
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in label_to_indices.items():
        # Calculate proportional split for this label
        label_splits = np.array_split(indices, num_clients)
        for i, split in enumerate(label_splits):
            client_indices[i].extend(split[:sample_sizes[i]])

    # Save subsets for each client
    for i, indices in enumerate(client_indices):
        client_data = [dataset[idx] for idx in indices]
        images, labels = zip(*client_data)
        images = torch.stack(images)
        labels = torch.tensor(labels)

        torch.save((images, labels), os.path.join(output_dir, f"client_{i+1}/data.pt"))
        print(f"Client {i+1}: {len(indices)} samples")

if __name__ == "__main__":
    # Parameters
    output_dir = "../data/mnist_splits"

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)

    # Split and save the dataset
    split_mnist(mnist_dataset, args.num_clients, args.total_samples, output_dir)
