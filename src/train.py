import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.sha256Dataset import (
    SHA256Dataset,
    SHA256Step1Dataset,
)
from src.data.logicGatesDataset import IdentityDataset, ANDDataset, ShiftRight_Dataset
from src.model.mlp import MLP


def train():
    # Parameters
    input_size = 512  # Max input string length
    # output_size = 256  # SHA-256 hash output is 32 bytes
    output_size = 256
    batch_size = 256
    epochs = 500
    learning_rate = 0.001

    # Dataset and DataLoader
    dataset = SHA256Step1Dataset(reverse=True)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_size = int(0.9 * len(dataset))  # 90% for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model =
    # Model, Loss, Optimizer
    # hidden_sizes = [256, 256, 256]
    hidden_sizes = [2048, 2048, 2048]
    # hidden_sizes = [64, 64, 64, 128, 256, 128, 64, 64, 64]

    activation = "relu"
    # dropout = 0.2
    dropout = 0
    model = MLP(input_size, output_size, hidden_sizes, activation, dropout)
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()  # Set model to training mode
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            sum_accuracies = 0
            counter = 0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                rounded_tensor = torch.round(outputs)
                correct_predictions = torch.sum(rounded_tensor == targets)
                accuracy = correct_predictions.item() / targets.numel()
                # print( f"Correct predictions: {correct_predictions.item()} / {targets.numel()}")
                sum_accuracies += accuracy
                counter += 1
            print(f"Average accuracy: {sum_accuracies * 100 / counter:.2f}%")
    print("Training complete!")


if __name__ == "__main__":
    train()
