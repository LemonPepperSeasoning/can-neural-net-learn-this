import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Multi-Layer Perceptron (MLP) implementation with configurable parameters.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_sizes (list[int]): List containing sizes of hidden layers.
            activation (str): Activation function ('relu', 'tanh', 'sigmoid').
            dropout (float): Dropout rate (default: 0.0).
        """
        super(MLP, self).__init__()

        # Map string to PyTorch activation function
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}'")

        self.activation = activation_map[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Create a list of layers
        layer_sizes = [input_size] + hidden_sizes
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(self.activation)
            if self.dropout:
                layers.append(self.dropout)

        # Final layer (output)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    input_size = 64
    output_size = 10
    hidden_sizes = [128, 256, 128]
    activation = "relu"
    dropout = 0.2

    mlp = MLP(input_size, output_size, hidden_sizes, activation, dropout)
    print(mlp)

    # Random input tensor
    x = torch.rand((5, input_size))
    output = mlp(x)
    print(output)
