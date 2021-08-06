import torch
from torch import nn
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE


# todo: continue implementing the train file. using
#  https://www.youtube.com/watch?v=MMkeLjcBTcI&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=9 at time 08:56
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader
