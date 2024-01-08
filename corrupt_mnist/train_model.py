import click
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from corrupt_mnist.models.model import MyNeuralNet
from corrupt_mnist.data.dataloader import mnist
from corrupt_mnist.visualizations.visualize import save_training_loss

#import pdb
#pdb.set_trace()

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-4, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print("Learning rate", lr)
    print("Epochs", epochs)

    model = MyNeuralNet()
    print(type(model))
    print(dir(model))
    print(isinstance(model, LightningModule))
    train_loader, test_loader = mnist()
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    trainer = Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)

    # TODO: log loss with wandb


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_loader = mnist()
    running_sum = 0
    pbar = tqdm(test_loader, total=len(test_loader))
    for images, labels in pbar:
        output = model(images)
        _, preds = torch.max(output, 1)
        running_sum += torch.sum(preds == labels.data)
        accuracy = torch.sum(preds == labels.data) / len(labels)
        pbar.set_description(f"Batch accuracy: {accuracy.item()*100:.2f}%")
    else:
        acc = running_sum / len(test_loader.dataset)
        print(f"Accuracy: {acc*100:.2f}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
