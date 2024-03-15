from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import load_presaved_model, load_untrained_model
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap


def perform_pca(device, seed, epoch, dataset, filename, pretrained=True):
    model = load_presaved_model(device, seed, epoch) if pretrained else load_untrained_model(device, seed)

    X = []
    for sample in dataset:
        img = sample['image'].numpy()  # Convert PyTorch tensor to numpy array
        X.append(img.flatten())

    X = np.array(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    predictions = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            img = sample['image'].unsqueeze(0).to(device)  # Convert to tensor and move to device
            output = model(img)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    num_classes = 6  # Assuming you have 6 classes

    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    # Color coding by label
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(num_classes):
        plt.scatter(X_pca[np.array(predictions) == i, 0], X_pca[np.array(predictions) == i, 1], c=colors[i], label=str(i))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot of Validation Set')
    plt.legend(title='Label')

    # Save the plot as a PNG file
    plt.savefig(filename)



def perform_pca_2(device, seed, epoch, dataset, filename, pretrained=True):
    plt.figure(figsize=(16, 8))
    model_pretrained, crit, optimizer = load_presaved_model(device, seed, epoch)
    model_untrained = load_untrained_model(device, seed)

    for m, model in enumerate([model_untrained, model_pretrained], start=1):
        X = []
        for sample in dataset:
            img = sample['image'].numpy()  # Convert PyTorch tensor to numpy array
            X.append(img.flatten())

        X = np.array(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        predictions = []
        model.eval()
        with torch.no_grad():
            for sample in dataset:
                img = sample['image'].unsqueeze(0).to(device)  # Convert to tensor and move to device
                output = model(img)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())

        num_classes = 6  # Assuming you have 6 classes

        # Create a scatter plot
        plt.subplot(1, 2, m)
        # Color coding by label
        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for i in range(num_classes):
            plt.scatter(X_pca[np.array(predictions) == i, 0], X_pca[np.array(predictions) == i, 1], c=colors[i], label=str(i))

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA Scatter Plot of Validation Set: {"Untrained" if m == 1 else "Pretrained"}')
        plt.legend(title='Label')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(filename)
