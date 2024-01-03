import matplotlib.pyplot as plt

def save_training_loss(training_loss):
    plt.figure()
    plt.title('Training Loss')
    plt.plot(training_loss)
    plt.savefig('reports/figures/training_loss.png')