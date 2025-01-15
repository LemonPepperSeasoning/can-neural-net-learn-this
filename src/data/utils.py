import os
import torch

DEFAULT_INPUT_BITS_SIZE = 256
DEFAULT_INPUT_BYTE_SIZE = DEFAULT_INPUT_BITS_SIZE // 8
SHA256_BITS_SIZE = 256

DATALOADER_SIZE = 256 * 256

HALF_INPUT_SIZE = DEFAULT_INPUT_BITS_SIZE // 8 // 2

ZEROS = b"\x00"
ONES = b"\xff"  # 0xff is 11111111 in binary
DEFAULT_MASK_BITS = ZEROS * HALF_INPUT_SIZE + ONES * HALF_INPUT_SIZE


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
