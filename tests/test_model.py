from corrupt_mnist.models.model import MyNeuralNet
import torch

def test_model():
    model = MyNeuralNet()
    # create random image with shape [1,28,28]
    image = torch.randn(1,28,28)
    # pass image through model
    output = model(image)
    # check that output is of shape [1,10]
    assert output.shape == (1,10)