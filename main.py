"""
End-to-end deep neural network compression pipeline.

Stages:
    1. Baseline training of a CompressionMLP on frozen AlexNet features (CIFAR-10)
    2. Magnitude (L1) pruning to 90% sparsity + short fine-tuning
    3. K-Means weight quantization (16 clusters)
    4. Huffman encoding of the quantized indices
    5. Compact .npz serialization of the compressed model
"""

import os
import torch

from models.model_cifar import CompressionMLP
from data.data_loader import CIFAR10_loader
from utils.training import train_and_eval
from compression.pruning import prune_model
from compression.quantization import quantize_model
from compression.huffman import huffman_encode_model, print_compression_summary
from utils.loading import save_model_npz
from config import config_device


def main():
    device = config_device()
    train_loader, test_loader = CIFAR10_loader()

    # 1. Baseline Model
    print("\n--- Training Baseline MLP ---")
    model = CompressionMLP().to(device)
    train_and_eval(model, train_loader, test_loader, device, epochs=5)

    # 2. Pruning
    print("\n--- Applying Pruning (90%) ---")
    prune_model(model, 0.90)
    print("Fine-tuning pruned model...")
    train_and_eval(model, train_loader, test_loader, device, epochs=2)

    # 3. Quantization
    print("\n--- Applying Quantization (16 clusters) ---")
    quantize_model(model, 16)

    # 4. Huffman Encoding
    print("\n--- Applying Huffman Encoding ---")
    huffman_encode_model(model)
    print_compression_summary(model)

    # 5. Serialization
    print("\n--- Serializing Model to NPZ ---")
    npz_path = os.path.join("compressed_models", "compressed_mlp.npz")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)  # ensure output dir exists
    save_model_npz(model, npz_path)


if __name__ == "__main__":
    main()
