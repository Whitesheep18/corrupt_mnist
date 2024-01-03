import click
import torch
from corrupt_mnist.models.model import MyNeuralNet
from data.dataloader import mnist
from visualizations.visualize import save_training_loss
from tqdm import tqdm


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
    print('Learning rate', lr)
    print('Epochs', epochs)
    
    # TODO: Implement training loop here
    model = MyNeuralNet()
    train_loader, _ = mnist()
    training_loss = []
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        epoch_loss = []
        for images, labels in pbar:
            optimizer.zero_grad()
            output = model(images)
            batch_loss = criterion(output, labels)
            epoch_loss.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}, Loss: {batch_loss.item()}")
        else:
            training_loss.append(sum(epoch_loss) / len(epoch_loss)) #avg
            torch.save(model, "models/model.pth")

        # save visualization of training
        save_training_loss(training_loss)
        

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
