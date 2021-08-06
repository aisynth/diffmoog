import torch
from torch import nn
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target_parameters in data_loader:
        input, target_parameters = input.to(device), target_parameters.to(device)

        # calculate loss
        predicted_parameters = model(input)
        loss = loss_fn(predicted_parameters, target_parameters)

        # backpropogate error and update wights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("--------------------------------------")
    print("Finished training")
