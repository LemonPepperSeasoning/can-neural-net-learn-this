import torch
from torch.utils.data import Dataset
from src.data.abstract_binary_dataset import AbstractBinaryDataset
from src.data.utils import (
    bytes_to_binary_tensor,
    get_random_bytes,
    convert_bytes_to_binary_str_representation,
    circular_right_shift,
    DEFAULT_INPUT_BITS_SIZE,
    DEFAULT_DATALOADER_SIZE,
    DEFAULT_MASK_BITS,
    SHIFT_N_BITS,
)


class IdentityFunctionDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: same random bits. (ie. input == target)

    Aim: Test if NN can return identity function
    """

    def __init__(self, input_bits_size=DEFAULT_INPUT_BITS_SIZE, reverse=False):
        super().__init__(input_bits_size, reverse=reverse)

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        return input_bytes


class ANDGateDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of AND operation on random bits and a mask bits

    Aim: Test if NN can learn basic AND operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        super().__init__(input_bits_size, reverse=reverse)
        self.mask_bits = mask_bits

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        result: bytes = bytes([b1 & b2 for b1, b2 in zip(input_bytes, self.mask_bits)])
        return result


class ORGateDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of OR operation on random bits and a mask bits

    Aim: Test if NN can learn basic OR operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        super().__init__(input_bits_size, reverse=reverse)
        self.mask_bits = mask_bits

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        result: bytes = bytes([b1 | b2 for b1, b2 in zip(input_bytes, self.mask_bits)])
        return result


class NOTGateDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of NOT operation on random bits

    Aim: Test if NN can learn basic NOT operation
    """

    B_255 = 0xFF

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
    ):
        super().__init__(input_bits_size, reverse=reverse)

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        result: bytes = bytes(~byte & NOTGateDataset.B_255 for byte in input_bytes)
        return result


class XORGateDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of XOR operation on random bits and a mask bits

    Aim: Test if NN can learn basic XOR operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
        mask_bits: bytes = DEFAULT_MASK_BITS,
    ):
        super().__init__(input_bits_size, reverse=reverse)
        self.mask_bits = mask_bits

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        result: bytes = bytes([b1 ^ b2 for b1, b2 in zip(input_bytes, self.mask_bits)])
        return result


class RightShiftFunctionDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of RIGHT SHIFT operation on random bits

    Aim: Test if NN can learn basic RIGHT SHIFT operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
        shift_n_bits=SHIFT_N_BITS,
    ):
        super().__init__(input_bits_size, reverse=reverse)
        self.shift_n_bits = shift_n_bits

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        byte_in_integer = int.from_bytes(input_bytes, byteorder="big")
        shifted_integer = byte_in_integer >> self.shift_n_bits
        shifted_bytes = shifted_integer.to_bytes(
            self.input_byte_size, byteorder="big", signed=False
        )
        return shifted_bytes


class CircularRightShiftFunctionDataset(AbstractBinaryDataset):
    """
    Input: random bits
    Target: result of RIGHT SHIFT operation on random bits

    Aim: Test if NN can learn basic RIGHT SHIFT operation
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
        shift_n_bits=SHIFT_N_BITS,
    ):
        super().__init__(input_bits_size, reverse=reverse)
        self.shift_n_bits = shift_n_bits

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        byte_in_integer = int.from_bytes(input_bytes, byteorder="big")
        shifted_integer = circular_right_shift(
            byte_in_integer, self.shift_n_bits, self.input_bits_size
        )
        shifted_bytes = shifted_integer.to_bytes(
            self.input_byte_size, byteorder="big", signed=False
        )
        return shifted_bytes


class Sigma0Dataset(AbstractBinaryDataset):
    """
    ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)
    """

    def __init__(
        self,
        input_bits_size=DEFAULT_INPUT_BITS_SIZE,
        reverse=False,
    ):
        super().__init__(input_bits_size, reverse=reverse)

    def apply_transformation(self, input_bytes: bytes) -> bytes:
        byte_in_integer = int.from_bytes(input_bytes, byteorder="big")

        rotr7 = circular_right_shift(byte_in_integer, 7, self.input_bits_size)
        rotr18 = circular_right_shift(byte_in_integer, 18, self.input_bits_size)
        shr3 = byte_in_integer >> 3

        rotr7_bytes = rotr7.to_bytes(
            self.input_byte_size, byteorder="big", signed=False
        )
        rotr18_bytes = rotr18.to_bytes(
            self.input_byte_size, byteorder="big", signed=False
        )
        shr3_bytes = shr3.to_bytes(self.input_byte_size, byteorder="big", signed=False)

        rotr7_xor_rotr18: bytes = bytes(
            [b1 ^ b2 for b1, b2 in zip(rotr7_bytes, rotr18_bytes)]
        )
        result: bytes = bytes([b1 ^ b2 for b1, b2 in zip(rotr7_xor_rotr18, shr3_bytes)])
        return result


if __name__ == "__main__":
    x = CircularRightShiftFunctionDataset()

    a, b = x[0]
    print(a)
    print(b)
