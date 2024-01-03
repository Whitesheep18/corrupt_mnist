import torch
import  click

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('imgs_path', type=click.Path(exists=True))
def main(model_path, imgs_path):
    model = torch.load(model_path)
    imgs = torch.load(imgs_path)
    dataset = ImageDataset(imgs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    predictions = predict(model, dataloader)
    print(len(predictions))
    print(predictions.argmax(dim=1))

if __name__ == '__main__':
    main()
