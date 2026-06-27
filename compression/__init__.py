"""Compression utilities: pruning, k-means quantization, Huffman coding."""
from .linear import ModifiedLinear
from .conv2d import ModifiedConv2d
from .pruning import prune_model
from .quantization import quantize_model
from .huffman import huffman_encode_model, print_compression_summary, get_compressed_size_bits
