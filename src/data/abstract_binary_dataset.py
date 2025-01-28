from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from src.data.utils import (
    bytes_to_binary_tensor,
    get_random_bytes,
    DEFAULT_INPUT_BITS_SIZE,
    DEFAULT_DATALOADER_SIZE,
)


class AbstractBinaryDataset(ABC, Dataset):
    """Wrapper around torch.utils.data.Dataset to provide a common interface for all binary datasets."""

    def __init__(
        self,
        input_bits_size: int = DEFAULT_INPUT_BITS_SIZE,
        dataset_size: int = DEFAULT_DATALOADER_SIZE,
        reverse=False,
    ):
        self.input_byte_size = input_bits_size // 8
        self.reverse = reverse
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        random_bytes: bytes = get_random_bytes(self.input_byte_size)
        target: bytes = self.apply_transformation(random_bytes)

        if self.reverse:
            return bytes_to_binary_tensor(target), bytes_to_binary_tensor(random_bytes)
        return bytes_to_binary_tensor(random_bytes), bytes_to_binary_tensor(target)

    @abstractmethod
    def apply_transformation(self, input_bytes: bytes) -> bytes:
        """
        apply transofrmation to the random bits
        """
        pass
