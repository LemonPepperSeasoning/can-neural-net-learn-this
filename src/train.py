import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.sha256Dataset import SHA256EncryptionDataset, SHA256DecryptionDataset
from src.data.logicGatesDataset import IdentityDataset
from src.model.mlp import MLP


def train():
    NUM_EPOCHS = 10

    # Parameters
    input_size = 256  # Max input string length
    output_size = 256  # SHA-256 hash output is 32 bytes
    batch_size = 256
    epochs = 500
    learning_rate = 0.001

    # Dataset and DataLoader
    dataset = IdentityDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # model =
    # Model, Loss, Optimizer
    hidden_sizes = [256, 256, 256]
    activation = "relu"
    dropout = 0.2
    model = MLP(input_size, output_size, hidden_sizes, activation, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print(targets)
    print(outputs)
    print("Training complete!")


if __name__ == "__main__":
    train()
