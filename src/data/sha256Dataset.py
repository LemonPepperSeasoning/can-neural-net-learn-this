import hashlib
import torch
from torch.utils.data import Dataset
from src.data.utils import (
    bytes_to_binary_tensor,
    get_random_bytes,
    DEFAULT_INPUT_BITS_SIZE,
    DATALOADER_SIZE,
)


class SHA256EncryptionDataset(Dataset):
    """
    Input: random bits
    Target: sha256 value of the random bits

    Aim: To train a model to predict the sha256 value of the random bits
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        sha256_value: bytes = hashlib.sha256(random_bits).digest()
        return bytes_to_binary_tensor(random_bits), bytes_to_binary_tensor(sha256_value)


class SHA256DecryptionDataset(Dataset):
    """
    Input: sha256 value of random bits
    Target: random bits

    Aim: To train a model to predict the random bits, given sha256 value
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        sha256_value: bytes = hashlib.sha256(random_bits).digest()
        return bytes_to_binary_tensor(sha256_value), bytes_to_binary_tensor(random_bits)


if __name__ == "__main__":
    x = SHA256EncryptionDataset()
    y = x[0]
    # print(y)
