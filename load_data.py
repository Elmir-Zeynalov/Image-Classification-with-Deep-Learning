import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class ImageDataset(data.Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.training_file = "train_set.txt"
        self.validation_file = "validation_set.txt"
        self.test_file = "test_set.txt"
        self.train    = train
        self.class_to_idx = {}
        self.image_filenames = []
        self.labels = []
        self.transform = transform

        with open(self.training_file if self.train else self.validation_file, 'r') as file:
            for line in file:
                parts = line.split(" ")
                self.image_filenames.append(parts[0])
                self.labels.append(parts[1].strip())

        self.class_to_idx = self.convert_label_to_ix(self.labels)
        print(self.class_to_idx)

        print("Done iterating through files")
        print(len(self.labels))

    def __len__(self):
        """
        :return: The total number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_path  = self.image_filenames[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = self.class_to_idx[self.labels[index]]

        label = torch.tensor(label, dtype=torch.long)
        sample = {'image': img, 'label': label, 'filepath': self.image_filenames[index]}

        return sample

    def convert_label_to_ix(self, label_list):
        classes = list(set(label_list))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return class_to_idx


'''
Class that iterates through the directories and creates files for the different datasets.
With the help of the sklearn library data is split into 3 sets (training, validation and test).
The split data is stored in each its own respective file.
In those files, the datapath to each image and the coresponding label is stored.
The stucture is the following:
    data_path_to_image.png label
    data_path_to_image.png label
    data_path_to_image.png label
    data_path_to_image.png label
    ...
Later on, the torch.Datasets will go to these files and fetch the path to the images.
This is done to avoid loading all the images in memory all at once.
'''
class DataSplitter():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.validation_size = 2000
        self.test_size = 3000
        self.files, self.labels = self.load_files(self.root_dir)
        self.create_dataset(self.files, self.labels)

    def load_files(self, root):
        files = []
        labels = []
        for sub_dir in os.listdir(root):
            sub_dir_path = os.path.join(root, sub_dir)
            if os.path.isdir(sub_dir_path):
                files_in_subdir = [
                    f
                    for f in os.listdir(sub_dir_path)
                    if os.path.isfile(os.path.join(sub_dir_path, f))
                ]
                for file in files_in_subdir:
                    files.append(os.path.join(sub_dir_path, file))
                    labels.append(sub_dir_path.split("/")[-1])
        return files, labels

    def save_file_paths(self, file_paths, labels, file_path):
        with open(file_path, "w") as f:
            for path, label in zip(file_paths, labels):
                f.write(f"{path} {label}\n")

    def create_dataset(self, files, labels):
        X_train_temp, X_remaining, y_train_temp, y_remaining = train_test_split(files, labels, test_size=(self.validation_size + self.test_size), stratify=labels, random_state=42)
        X_validation, X_test, y_validation, y_test = train_test_split(X_remaining, y_remaining, test_size=(self.test_size / (self.validation_size + self.test_size)), stratify=y_remaining, random_state=42)

        self.save_file_paths(X_train_temp, y_train_temp, "train_set.txt")
        self.save_file_paths(X_validation, y_validation, "validation_set.txt")
        self.save_file_paths(X_test, y_test, "test_set.txt")


def run_epoch_train(model, criterion, optimizer, epoch, data_loader):
    model.train()
    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list      = [0,1,2,3,4,5]

    for idx, batch in enumerate(data_loader):
        images = batch['image'].to('cuda') # send data to GPU
        labels = batch['label'].to('cuda') # send data to GPU
    
        prediction = model(images)
        loss        = criterion(prediction, labels)
        loss_numpy = loss.detach().cpu().numpy()
        total_loss += loss_numpy
        #total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the number of correct classifications and the confusion matrix
        predicted_label  = prediction.max(1, keepdim=True)[1][:,0]
        correct          += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)

        
        #print(f'Epoch={epoch} | {(idx+1)/len(data_loader)*100:.2f}% | loss = {loss:.5f}')

    loss_avg         = total_loss / len(data_loader)
    accuracy         = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)

    return loss_avg, accuracy, confusion_matrix

def train(model, optimizer, criterion, num_epochs, data_loader):
#    print("---Training---")
#    model.train()
#    for epoch in range(num_epochs):
#        for batch in data_loader:
#            inputs = batch['image']
#            labels = batch['label']
#
#            optimizer.zero_grad()
#
#            inputs, labels = inputs.to(device), labels.to(device)
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#            loss.backward()
#            optimizer.step()
#        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    correct = 0
    total_samples = 0
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list= [0,1,2,3,4,5]
    config = {'epochs': num_epochs}

    train_loss   = np.zeros(shape=config['epochs'])
    train_acc    = np.zeros(shape=config['epochs'])
    train_confusion_matrix   = np.zeros(shape=(6,6,config['epochs']))
    print("---Training---")
    
    for epoch in range(num_epochs):
        train_loss[epoch], train_acc[epoch], train_confusion_matrix[:,:,epoch]     = \
                            run_epoch_train(model, criterion, optimizer, epoch, data_loader)
    
    print("Training losses")
    print(train_loss)

    print("********")
    ind = np.argmax(train_acc)
    class_accuracy = train_confusion_matrix[:,:,ind]
    classes = [0,1,2,3,4,5]

    accuracy_per_class = []
    precision_per_class = []

    for ii in range(len(classes)):
        true_positives = train_confusion_matrix[ii, ii, ind]
        false_positives = np.sum(train_confusion_matrix[:, ii, ind]) - true_positives
        precision = true_positives / (true_positives + false_positives)
        precision_per_class.append(precision)

        acc = train_confusion_matrix[ii,ii,ind] / np.sum(train_confusion_matrix[ii,:,ind])
        accuracy_per_class.append(acc)

        print(f'Class {str(classes[ii]).ljust(5)}: Accuracy={acc*100:.01f}%, Precision={precision*100:.01f}%')
    average_accuracy = np.mean(accuracy_per_class)
    average_precision = np.mean(precision_per_class)
    print("****Averages****")
    print(f'Average Accuracy: {average_accuracy*100:.01f}%')
    print(f'Average Precision: {average_precision*100:.01f}%')
    return train_loss



def run_epoch_eval(model, criterion, epoch, data_loader):
    model.eval()

    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list      = [0,1,2,3,4,5]

    for idx, batch in enumerate(data_loader):
        images = batch['image'].to('cuda') # send data to GPU
        labels = batch['label'].to('cuda') # send data to GPU
        
        with torch.no_grad():
            prediction = model(images)
            loss        = criterion(prediction, labels)
            loss_numpy = loss.detach().cpu().numpy()
            total_loss += loss_numpy
            
        # Update the number of correct classifications and the confusion matrix
        predicted_label  = prediction.max(1, keepdim=True)[1][:,0]
        correct          += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)

    loss_avg         = total_loss / len(data_loader)
    accuracy         = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)
    print("LOSS_AVG:", loss_avg)
    return loss_avg, accuracy, confusion_matrix


def validate(model, criterion, num_epochs, data_loader):
    correct = 0
    total_samples = 0
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list= [0,1,2,3,4,5]
    config = {'epochs': num_epochs}


    val_loss   = np.zeros(shape=config['epochs'])
    val_acc    = np.zeros(shape=config['epochs'])
    val_confusion_matrix   = np.zeros(shape=(6,6,config['epochs']))


    print("---Validating---")
    
    for epoch in range(num_epochs):
        #print("epooooo")
        val_loss[epoch], val_acc[epoch], val_confusion_matrix[:,:,epoch]     = \
                            run_epoch_eval(model, criterion,  epoch, data_loader)
        #print("")
    
    print("Validation losses")
    print(val_loss)

    print("********")
    ind = np.argmax(val_acc)
    class_accuracy = val_confusion_matrix[:,:,ind]
    classes = [0,1,2,3,4,5]

    accuracy_per_class = []
    precision_per_class = []

    for ii in range(len(classes)):
        true_positives = val_confusion_matrix[ii, ii, ind]
        false_positives = np.sum(val_confusion_matrix[:, ii, ind]) - true_positives
        precision = true_positives / (true_positives + false_positives)
        precision_per_class.append(precision)

        acc = val_confusion_matrix[ii,ii,ind] / np.sum(val_confusion_matrix[ii,:,ind])
        accuracy_per_class.append(acc)

        print(f'Class {str(classes[ii]).ljust(5)}: Accuracy={acc*100:.01f}%, Precision={precision*100:.01f}%')
    average_accuracy = np.mean(accuracy_per_class)
    average_precision = np.mean(precision_per_class)
    print("****Averages****")
    print(f'Average Accuracy: {average_accuracy*100:.01f}%')
    print(f'Average Precision: {average_precision*100:.01f}%')
    return val_loss


def plot_losses(val_loss, train_loss, save_path):
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(2, 1, 1)
    ax.plot(train_loss, 'b', label='train loss')
    ax.plot(val_loss, 'r', label='validation loss')
    ax.grid()
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    ax.legend(loc='upper right', fontsize=16)

 
    plt.savefig(save_path)  
    plt.close()

    

def train_resnet18_model(device, train_loader, val_loader):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = models.resnet18(pretrained=True)
    num_classes = 6 #len(dataset.class_to_idx)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)  # Move the model to GPU or CPU

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    train_losses = train(model, optimizer, criterion, num_epochs, train_loader)
    print()
    val_loss = validate(model, criterion, num_epochs, val_loader)
    plot_losses(val_loss, train_losses, "validation_loses.png")


def create_datasets_and_loaders(root_path, transform=None, seed=50):

    def worker_init_fn(worker_id):
        torch.manual_seed(seed + worker_id)
    
    # Training dataset and dataloader
    train_dataset = ImageDataset(root_dir=root_path, train=True, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4,worker_init_fn=worker_init_fn)

    # Validation dataset and dataloader
    val_dataset = ImageDataset(root_dir=root_path, train=False, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    return train_loader, val_loader

if __name__ == "__main__":
    root_path = "Images/mandatory1_data/"

    #split data into training, validation and test data sets
    #dataset = DataSplitter(root_path)
    print("Done creating train, validation and test sets...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])

    # a bit better
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])

    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
#
#
    #transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.GaussianBlur(kernel_size=3),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])
#
    #transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])

    #trash
    #transform = transforms.Compose([
    #    transforms.RandomResizedCrop(224),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.RandomRotation(degrees=30),
    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])


    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data sets and loaders
    train_loader, val_loader = create_datasets_and_loaders(root_path=root_path, transform=transform, seed=seed)

    #training and validation
    train_resnet18_model(device, train_loader, val_loader)