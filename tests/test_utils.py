import pytest
from src.data.utils import (
    circular_right_shift,
)


@pytest.mark.parametrize(
    "value, shift, bit_size, expected",
    [
        (0b10110011, 2, 8, 0b11101100),  # Basic shift
        (0b11001001, 3, 8, 0b00111001),  # Basic shift
        (0b10110011, 0, 8, 0b10110011),  # No shift
        (0b10110011, 8, 8, 0b10110011),  # Full rotation
        (0b11001001, 8, 8, 0b11001001),  # Shift equals bit size
        (0b10110011, 10, 8, 0b11101100),  # Large shift (10 % 8 = 2)
        (0b00000000, 4, 8, 0b00000000),  # Minimum value
        (0xFF, 4, 8, 0xFF),  # Maximum value (all 1s remain unchanged)
        (
            0b1010101010101010,
            4,
            16,
            0b1010101010101010 >> 4 | (0b1010 << 12),
        ),  # 16-bit test
    ],
)
def test_circular_right_shift(value, shift, bit_size, expected):
    assert circular_right_shift(value, shift, bit_size) == expected
