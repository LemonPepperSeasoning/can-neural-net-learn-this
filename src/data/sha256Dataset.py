import hashlib
from src.data.utils import (
    int_to_bytes,
    DEFAULT_INPUT_BITS_SIZE,
)
from src.data.abstract_binary_dataset import AbstractBinaryDataset


class SHA256Dataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: sha256 value of the random bits

    Aim: To train a model to predict the sha256 value of the random bits
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE, reverse=False):
        super().__init__(input_bits_size, reverse=reverse)

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        return hashlib.sha256(input_bytes).digest()


class SHA256Step1Dataset(AbstractBinaryDataset):
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

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE, reverse=False):
        super().__init__(input_bits_size, reverse=reverse)

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        message_length = len(input_bytes) * 8
        message_length_in_bytes = int_to_bytes(message_length, 8)

        padding_length = (512 - (message_length + 64) % 512) // 8
        padding = b"\x80" + b"\x00" * (padding_length - 1)

        padded_message: bytes = input_bytes + padding + message_length_in_bytes
        return padded_message


if __name__ == "__main__":
    x = SHA256EncryptionDataset()
    y = x[0]
    # print(y)
