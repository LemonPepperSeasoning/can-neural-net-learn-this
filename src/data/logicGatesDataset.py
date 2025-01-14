import torch
from torch.utils.data import Dataset
from src.data.utils import (
    binary_str_to_tensor,
    convert_bytes_to_binary_str_representation,
    get_random_bytes,
    DEFAULT_INPUT_BITS_SIZE,
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
        # raise NotImplementedError("Dataset has no fixed length")
        return 256 * 256

    def __getitem__(self, idx):
        random_bits: bytes = get_random_bytes(self.input_byte_size)
        random_bits_str: str = convert_bytes_to_binary_str_representation(random_bits)
        input: torch.Tensor = binary_str_to_tensor(random_bits_str)
        return input, input


class ANDDataset(Dataset):
    pass


class ORDataset(Dataset):
    pass


class NOTDataset(Dataset):
    pass


class XOR_Dataset(Dataset):
    pass


class CombinedLogicGatesDataset(Dataset):
    pass
