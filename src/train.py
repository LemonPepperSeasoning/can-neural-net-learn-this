import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_source.utils import SHA256Dataset
from src.model.mlp import MLP


def train():
    NUM_EPOCHS = 10

    # Parameters
    input_size = 256  # Max input string length
    output_size = 256  # SHA-256 hash output is 32 bytes
    dataset_size = 1000
    batch_size = 32
    epochs = 5
    learning_rate = 0.001

    # Dataset and DataLoader
    dataset = SHA256Dataset(dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model =
    # Model, Loss, Optimizer
    hidden_sizes = [128, 256, 128]
    activation = "relu"
    dropout = 0.2
    model = MLP(input_size, output_size, hidden_sizes, activation, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Pad or truncate inputs to a fixed size
            padded_inputs = torch.zeros((inputs.size(0), input_size))
            padded_inputs[:, : inputs.size(1)] = inputs[:, :input_size]

            # Forward pass
            outputs = model(padded_inputs)
            print(outputs)

            print(outputs.shape)

            print(inputs.shape)
            print(targets.shape)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train()
