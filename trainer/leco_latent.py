import torch
import torchvision
import numpy as np
from modules import scripts

EXT_DIR = scripts.basedir()

class DummyLatent():
    def __init__(self, height, width):
        self.height, self.width = height, width
        downloaded = torchvision.datasets.CIFAR10(f"{EXT_DIR}/cifar10", download=True)
        self.cifar10 = np.transpose(downloaded.data, (0, 3, 1, 2))  # (50000, 3, 32, 32)
        B, C, H, W = self.cifar10.shape
        self.cifar10 = self.cifar10.reshape(B*C, H, W)  # Tensor([150000, 32, 32])

    def get_batch(self, batch_size):
        B, _, _ = self.cifar10.shape
        index_batch = np.random.randint(0, B, (batch_size, 4))
        batch = np.stack([self.cifar10[index] for index in index_batch])  # ndarray([batch_size, 4, 32, 32])
        batch = torch.from_numpy(batch)
        batch = torch.nn.functional.interpolate(batch, size=(self.height, self.width), mode='bicubic')
        batch = (batch / torch.max(batch) - 0.5) * 2.0
        return batch