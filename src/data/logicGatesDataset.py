import torch
from torch.utils.data import Dataset
from src.data.utils import (
    binary_str_to_tensor,
    convert_bytes_to_binary_str_representation,
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
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)
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
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)

        result: bytes = bytes([b1 & b2 for b1, b2 in zip(random_bits, self.mask_bits)])
        result_in_str: str = convert_bytes_to_binary_str_representation(result)
        target: torch.Tensor = binary_str_to_tensor(result_in_str)
        return input, target


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
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)

        result: bytes = bytes([b1 | b2 for b1, b2 in zip(random_bits, self.mask_bits)])
        result_in_str: str = convert_bytes_to_binary_str_representation(result)
        target: torch.Tensor = binary_str_to_tensor(result_in_str)
        return input, target


class NOTDataset(Dataset):
    pass


class XOR_Dataset(Dataset):
    pass


class CombinedLogicGatesDataset(Dataset):
    pass


if __name__ == "__main__":
    x = ANDDataset()

    a, b = x[0]
    print(a)
    print(b)

    bit_string = "".join(f"{byte:08b}" for byte in b)
    print("Bits:", bit_string)

    print(a.__class__)
    print(b.__class__)
    print(len(b))
