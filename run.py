import os
import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms, models
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, average_precision_score
from Utilities.utils import create_folder, create_dir, save_model, load_model, create_resnet50_model, load_presaved_model
from Utilities.feature_map_utils import feature_map_statistics, analyze_feature_maps
from Utilities.pca_analytics import perform_pca
from Utilities.data_utils import ImageDataset, DataSplitter, create_datasets_and_loaders
import shutil
from Utilities.plot_utils import plot_losses_and_accuracies, plot_precision_and_accuracy 
from Utilities.accuracy_precision_utils import calc_accuracy_per_class, accuracies_for_all_epochs, calc_average_precision_per_class, average_precisions_mAPs_for_all_epochs
from Utilities.model_evaluation_utils import test_model_and_compare_softmaxes, evaluate
from Utilities.model_training_utils import train_resnet_model



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #The path below is where i fetch the Images from. It is only used once to create the datasets which will be stored in the Datasets directory
    #Make sure 
    root_path = "/itf-fi-ml/shared/courses/IN3310/mandatory1_data" #"Images/mandatory1_data/" 

    datasets_path = create_dir('Datasets')
    sotmax_path = create_dir('Softmaxes')
    graphs_path = create_dir('Graphs')
    
    #split data into training, validation and test data sets
    dataset = DataSplitter(root_path, datasets_path)

    # this transform is used for the training!
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  # New: Randomly flip the image vertically
        transforms.RandomRotation(degrees=30),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.2),  # New: Random affine transformation
        #transforms.RandomPerspective(),  # New: Random perspective transformation
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # New: Random erasing
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # New: Random Gaussian blur
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    num_epochs = 30 #20 #30
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
    #Task 1a-1b) data sets and loaders 
    print('\033[96m***[Data loading]***\033[0m')
    train_loader, val_loader, test_loader, val_dataset = create_datasets_and_loaders(root_path=root_path, datasets_path=datasets_path,transform=transform, seed=seed)

    # Task 1c-1e) training
    print('\033[96m***[Model Training]***\033[0m')
    mod = train_resnet_model(device, num_epochs, graphs_path, train_loader, val_loader, seed)

    #print("Evaluation on Test Set")
    #evaluate(mod, test_loader, "Randpath", device, seed, num_epochs, "Softmax_path", 0.0001)

    # Task 1F) Running model on test set and then comparing against saved softmaxes
    print('\033[96m***[Model Evaluation]***\033[0m')
    test_model_and_compare_softmaxes(test_loader, device, num_epochs, seed, datasets_path, sotmax_path, threshold=0.0001)

    # Task 2
    print('\033[96m***[Feature Map Statistics]***\033[0m')
    feature_map_statistics(test_loader, device, num_epochs, seed)

    # Task 3
    print('\033[96m***[Principal Component Analysis]***\033[0m')
    perform_pca(device, seed, num_epochs, val_dataset, os.path.join(graphs_path,"pca_epochs.png"))