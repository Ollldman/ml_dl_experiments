from h11 import Data
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader

from ml_dl_experiments.settings import settings


def load_mnist_to_batches(download_from_source: bool = False,
                         path_to_loading: str = './data/',
                         train_size_index: float = 0.8,
                         train_val_test: bool = False
                         ) -> tuple[DataLoader, DataLoader, DataLoader] | tuple[DataLoader, DataLoader]:
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.1307], std=[0.3081])
    ])

    train_data = MNIST(root=path_to_loading,
                        train=True,
                        download=download_from_source,
                        transform=transform)
    test_data  = MNIST(root=path_to_loading,
                        train=False,
                        download=download_from_source,
                        transform=transform)

    if train_val_test:
        train_size: int = int(train_size_index * len(train_data))
        val_size: int = len(train_data) - train_size

        train_data, val_data = random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_data,
                                batch_size=64,
                                shuffle=True)

        val_loader = DataLoader(val_data,
                                batch_size=64,
                                shuffle=False)
        
        test_loader = DataLoader(test_data,
                                batch_size=1000,
                                shuffle=False)
        images, labels = next(iter(train_loader))


        # Печатаем формы тензоров
        print("images.shape:", images.shape)   # torch.Size([64, 1, 28, 28])
        print("labels.shape:", labels.shape)   # torch.Size([64])


        # Выведем метку первого изображения
        first_label = labels[0].item()
        print("Метка первого изображения в батче:", first_label)
        return train_loader, val_loader, test_loader
    
    else:

        train_loader = DataLoader(train_data,
                                batch_size=64,
                                shuffle=True)
        
        test_loader = DataLoader(test_data,
                                batch_size=1000,
                                shuffle=False)

        images, labels = next(iter(train_loader))


        # Печатаем формы тензоров
        print("images.shape:", images.shape)   # torch.Size([64, 1, 28, 28])
        print("labels.shape:", labels.shape)   # torch.Size([64])


        # Выведем метку первого изображения
        first_label = labels[0].item()
        print("Метка первого изображения в батче:", first_label)
        return train_loader,  test_loader
