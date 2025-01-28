import hashlib
import torch
from torch.utils.data import Dataset
from src.data.utils import (
    bytes_to_binary_tensor,
    get_random_bytes,
    int_to_bytes,
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


class SHA256Step1Dataset(Dataset):
    """
    Step 1 of SHA256 encryption: Padding the input bits:
    Given N bits, transoform into N + Padding + 64 = n x 512

    For example:
    Given 256 bits, output will be
    Original 256 bits + 192 + 64 = 512 bits

    - the 192 bits will be all 0s except the first bit. ie, 1000...0
    - the 64 btis will be the original length of the input bits in binary

    Input: random bits (Default 128 bits)
    Target: sha256 value of the random bits (Default 512 bits)

    Aim: To train a model to predict the sha256 value of the random bits
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)

        message_length = len(random_bits) * 8
        message_length_in_bytes = int_to_bytes(message_length, 8)

        padding_length = (512 - (message_length + 64) % 512) // 8
        padding = b"\x80" + b"\x00" * (padding_length - 1)
        padded_message = random_bits + padding + message_length_in_bytes

        return (
            bytes_to_binary_tensor(random_bits),
            bytes_to_binary_tensor(padded_message),
        )


if __name__ == "__main__":
    x = SHA256EncryptionDataset()
    y = x[0]
    # print(y)
