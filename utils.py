import os
import torch
from torchvision import transforms, models


def NN_models(model, model_index, num_classes=6):

    models = [
        torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),  # Dropout layer with dropout probability 0.5
            torch.nn.Linear(model.fc.in_features, num_classes)),
        torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 1024),  # Increase the number of hidden units
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, num_classes)),
        torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 512),  
            torch.nn.BatchNorm1d(512),  # Add batch normalization layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, num_classes)),
        torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 512),  
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 256),  # Add another fully connected layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, num_classes)),
        torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 512),  
            torch.nn.LeakyReLU(inplace=True),  # Use Leaky ReLU activation function
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, num_classes)),
        torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, num_classes)),
    ]

    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.001),
        torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001),
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001),
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001),

        ## weight_decay
        torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.002),
        torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.003),
        torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.004),
        torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005),

        #lr=0.001 + weight_decay
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.002),
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003),
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.004),
        torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005),

        #lr=0.01 + weight_decay
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.002),
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003),
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.004),
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005),
    ]

    return models[model_index]


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_model(model, save_path):
    torch.save(model, save_path)

def load_model(path):
    model=torch.load(path)
    return model

def create_resnet50_model(device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = models.resnet50(pretrained=True)
    model.fc = NN_models(model, 3)
    model = model.to(device)  # Move the model to GPU or CPU

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    return model, optimizer, criterion

def load_untrained_model(device,seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = models.resnet50(pretrained=True)
    model.fc = NN_models(model, 3)
    model = model.to(device)  

    return model

def load_presaved_model(device, seed, epoch, load_path=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    last_epoch = epoch # last epoch

    if load_path is None:
        curr_dir = os.getcwd()
        load_path = f'{curr_dir}/SavedModels/model_checkpoint_epoch_{last_epoch}.pth'

    model = load_model(load_path)
    print(f'loaded: {load_path}')
    print('\033[92mModel load successful!\033[0m')
    model = model.to(device)
    return model