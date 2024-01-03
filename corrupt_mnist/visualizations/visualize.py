import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
def save_training_loss(training_loss):
    plt.figure()
    plt.title('Training Loss')
    plt.plot(training_loss)
    plt.savefig('reports/figures/training_loss.png')

def visualize_model_features(model_path):
    model = torch.load(model_path)
    weights = model.fc4.weight.data
    tsne = TSNE(n_components=2, random_state=0, perplexity=3)
    weights_2d = tsne.fit_transform(weights.numpy())
    print(weights_2d.shape)
    plt.figure()
    plt.scatter(weights_2d[:,0], weights_2d[:,1])
    # add labels
    for i, txt in enumerate(range(10)):
        plt.annotate(txt, (weights_2d[i,0], weights_2d[i,1]))
    plt.savefig('reports/figures/weights.png')


if __name__ == '__main__':
    visualize_model_features('models/model.pth')
    