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


class SHA256EncryptionDataset(Dataset):
    INPUT_BITS_SIZE = 256
    INPUT_BYTE_SIZE = INPUT_BITS_SIZE // 8
    SHA256_BYTE_SIZE = 256

    def __init__(self):
        pass

    def __len__(self):
        # raise NotImplementedError("Dataset has no fixed length")
        return 1000

    def __getitem__(self, idx):
        random_256bits = get_random_bytes(SHA256EncryptionDataset.INPUT_BYTE_SIZE)
        random_256bits_str = convert_bytes_to_binary_str_representation(random_256bits)
        input = binary_str_to_tensor(random_256bits_str)

        sha256_value = hashlib.sha256(random_256bits).digest()
        sha256_value_str = convert_bytes_to_binary_str_representation(sha256_value)
        output = binary_str_to_tensor(sha256_value_str)
        return input, output


class SHA256DecryptionDataset(Dataset):
    INPUT_BYTE_SIZE = 256
    SHA256_BYTE_SIZE = 256


if __name__ == "__main__":
    x = SHA256EncryptionDataset()
    y = x[0]
    # print(y)
