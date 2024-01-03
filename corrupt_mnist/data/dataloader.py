import torch
from torch.utils.data import Dataset, DataLoader
import os


class CorruptMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train:
            imgs = []
            labels = []
            chunks = len([x for x in os.listdir("data/raw/") if "train_images" in x])
            for i in range(chunks):
                img_set = torch.load(f"data/processed/train_images_{i}.pt")
                label_set = torch.load(f"data/processed/train_target_{i}.pt")
                imgs.append(img_set)
                labels.append(label_set)
            self.data = torch.cat(imgs, dim=0)
            self.labels = torch.cat(labels, dim=0)
        else:
            self.data = torch.load("data/processed/test_images.pt")
            self.labels = torch.load("data/processed/test_target.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        return image, label


def mnist(batch_size=64):
    """
    Loads the CorruptMNIST dataset and returns DataLoaders for the training and test sets.

    Args:
        batch_size (int, optional): The batch size to use for the DataLoader. Defaults to 64.

    Returns:
        tuple: A tuple containing the training and test DataLoaders.
    """
    trainset = CorruptMNIST(train=True)
    testset = CorruptMNIST(train=False)

    train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train, test


if __name__ == "__main__":
    train, test = mnist()
    images, labels = next(iter(train))
    print(images.shape)
