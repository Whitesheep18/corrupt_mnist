from corrupt_mnist.data.dataloader import CorruptMNIST, mnist
import numpy as np
from tests import _PATH_DATA
import os.path
import pytest
import torch

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/processed/test_images.pt"), reason="Data files not found")
def test_data():
    N_train = 10*5000
    N_test = 5000
    train_dataset = CorruptMNIST(train=True)
    test_dataset = CorruptMNIST(train=False)
    assert len(train_dataset) == N_train # train set did not have the expected number of samples
    assert len(test_dataset) == N_test # test set did not have the expected number of samples
    
    image = train_dataset[0][0]
    assert image.shape == torch.Size([28, 28]) # train image did not have the expected shape
    image = test_dataset[0][0]
    assert image.shape == torch.Size([28, 28]) # test image did not have the expected shape

    labels_in_train =  len(np.unique(train_dataset.labels))
    assert labels_in_train == 10 # train set did not have the expected number of labels


    labels_in_test =  len(np.unique(test_dataset.labels))
    assert labels_in_test == 10 # test set did not have the expected number of labels

@pytest.mark.parametrize("test_input,expected", [(64, 64), (128, 128)])
def test_dataloaders(test_input, expected):
    train_loader, test_loader = mnist(batch_size=test_input)

    imgs, labels = next(iter(train_loader))
    assert imgs.shape[0] == expected
    assert labels.shape[0] == expected

    imgs, labels = next(iter(test_loader))
    assert imgs.shape[0] == expected
    assert labels.shape[0] == expected

