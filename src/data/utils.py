import hashlib
import torch
from torch.utils.data import Dataset


# def sha256(data):
#     return sha256(data).hexdigest()


def incremental_integer():
    i = 0
    while True:
        yield i
        i += 1


def compute_sha256(input_string):
    hash_hex = hashlib.sha256(input_string.encode()).hexdigest()
    # Convert the hex hash to a list of integers (byte values)
    hash_bytes = bytes.fromhex(hash_hex)
    return (
        torch.tensor(list(hash_bytes), dtype=torch.float32) / 255.0
    )  # Normalize to [0, 1]


class SHA256Dataset(Dataset):
    INPUT_BYTE_SIZE = 256
    SHA256_BYTE_SIZE = 256

    def __init__(self, size):
        self.data = [str(i) for i in range(size)]
        self.labels = [compute_sha256(d) for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert string input into numerical input (padded to max_len)
        input = torch.tensor([ord(c) for c in self.data[idx]], dtype=torch.float32)
        if len(input) < SHA256Dataset.SHA256_BYTE_SIZE:
            padded_input = torch.zeros(SHA256Dataset.SHA256_BYTE_SIZE)
            padded_input[: len(input)] = input
        else:
            padded_input = input[: SHA256Dataset.SHA256_BYTE_SIZE]

        print(padded_input)
        target = self.labels[idx]
        return padded_input, target
