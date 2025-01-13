import os
import hashlib
import torch
from torch.utils.data import Dataset


def binary_str_to_tensor(string: str) -> torch.Tensor:
    """
    Convert a string of 0s and 1s to a tensor of 0s and 1s.
    """
    tensor = torch.tensor([int(char) for char in string], dtype=torch.float32)
    return tensor


def get_random_bytes(n_bytes: int) -> bytes:
    random_bytes = os.urandom(n_bytes)
    return random_bytes


def convert_bytes_to_binary_str_representation(data: bytes) -> str:
    """
    given b"Hello, World!" format
    return string made up of 0s and 1s. Eg: "0100101010011"
    """
    binary_representation = "".join(format(byte, "08b") for byte in data)
    return binary_representation


DEFAULT_INPUT_BITS_SIZE = 256
DEFAULT_INPUT_BYTE_SIZE = DEFAULT_INPUT_BITS_SIZE // 8
SHA256_BITS_SIZE = 256


class SHA256EncryptionDataset(Dataset):
    """
    Input: random bits
    Target: sha256 value of the random bits

    Aim: To train a model to predict the sha256 value of the random bits
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        # raise NotImplementedError("Dataset has no fixed length")
        return 1000

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)

        sha256_value: bytes = hashlib.sha256(random_bits).digest()
        sha256_value_str: str = convert_bytes_to_binary_str_representation(sha256_value)
        output: torch.Tensor = binary_str_to_tensor(sha256_value_str)
        return input, output


class SHA256DecryptionDataset(Dataset):
    """
    Input: sha256 value of random bits
    Target: random bits

    Aim: To train a model to predict the random bits, given sha256 value
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        # raise NotImplementedError("Dataset has no fixed length")
        return 1000

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)

        sha256_value: bytes = hashlib.sha256(random_bits).digest()
        sha256_value_str: str = convert_bytes_to_binary_str_representation(sha256_value)
        output: torch.Tensor = binary_str_to_tensor(sha256_value_str)
        return output, input


if __name__ == "__main__":
    x = SHA256EncryptionDataset()
    y = x[0]
    # print(y)
