import torch
from torch.utils.data import Dataset
from src.data.utils import (
    bytes_to_binary_tensor,
    get_random_bytes,
    DEFAULT_INPUT_BITS_SIZE,
    DATALOADER_SIZE,
    DEFAULT_MASK_BITS,
)


class IdentityDataset(Dataset):
    """
    Input: random bits
    Target: same random bits. (ie. input == target)

    Aim: Test if NN can return identity function
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        input: torch.Tensor = bytes_to_binary_tensor(random_bits)
        return input, input


class ANDDataset(Dataset):
    """
    Input: random bits
    Target: result of AND operation on random bits and a mask bits

    Aim: Test if NN can learn basic AND operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        self.input_byte_size = input_bits_size // 8
        self.mask_bits = mask_bits

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        result: bytes = bytes([b1 & b2 for b1, b2 in zip(random_bits, self.mask_bits)])
        return bytes_to_binary_tensor(random_bits), bytes_to_binary_tensor(result)


class ORDataset(Dataset):
    """
    Input: random bits
    Target: result of OR operation on random bits and a mask bits

    Aim: Test if NN can learn basic OR operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        self.input_byte_size = input_bits_size // 8
        self.mask_bits = mask_bits

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        result: bytes = bytes([b1 | b2 for b1, b2 in zip(random_bits, self.mask_bits)])
        return bytes_to_binary_tensor(random_bits), bytes_to_binary_tensor(result)


class NOTDataset(Dataset):
    """
    Input: random bits
    Target: result of NOT operation on random bits

    Aim: Test if NN can learn basic NOT operation
    """

    B_255 = 0xFF

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
    ):
        self.input_byte_size = input_bits_size // 8

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        result: bytes = bytes(~byte & NOTDataset.B_255 for byte in random_bits)
        return bytes_to_binary_tensor(random_bits), bytes_to_binary_tensor(result)


class XOR_Dataset(Dataset):
    """
    Input: random bits
    Target: result of XOR operation on random bits and a mask bits

    Aim: Test if NN can learn basic XOR operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        self.input_byte_size = input_bits_size // 8
        self.mask_bits = mask_bits

    def __len__(self):
        return DATALOADER_SIZE

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        result: bytes = bytes([b1 ^ b2 for b1, b2 in zip(random_bits, self.mask_bits)])
        return bytes_to_binary_tensor(random_bits), bytes_to_binary_tensor(result)


class CombinedLogicGatesDataset(Dataset):
    pass


if __name__ == "__main__":
    x = NOTDataset()

    a, b = x[0]
    print(a)
    print(b)
