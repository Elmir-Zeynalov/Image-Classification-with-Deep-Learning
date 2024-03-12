import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from utils import create_folder, save_model, load_model

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


def run_epoch(model, epoch, data_loader, optimizer, loss_fn, is_training):
    if is_training==True: 
        model.train()
    else:
        model.eval()

    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list      = [0,1,2,3,4,5]

    predicted = []
    true_values = [] 
    for batch_idx, data_batch in enumerate(data_loader):
        images = data_batch['image'].to('cuda') # send data to GPU
        labels = data_batch['label'].to('cuda') # send data to GPU

        if not is_training:
            with torch.no_grad():
                prediction = model(images)
                loss        = loss_fn(prediction, labels)
                total_loss += loss.item()
                print("EPOCH:", epoch , "batchid:", batch_idx, "loss:", loss)    
            
        elif is_training:
            prediction = model(images)
            loss        = loss_fn(prediction, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # Update the number of correct classifications and the confusion matrix
        predicted_label  = prediction.max(1, keepdim=True)[1][:,0]
        correct          += predicted_label.eq(labels).cpu().sum().numpy()
        confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)

        predicted.extend(predicted_label.cpu().numpy()) 
        true_values.extend(labels.cpu().numpy())


    loss_avg         = total_loss / len(data_loader)
    accuracy         = correct / len(data_loader.dataset)
    confusion_matrix = confusion_matrix / len(data_loader.dataset)

    return loss_avg, accuracy, confusion_matrix, predicted, true_values

def run_the_training(model, epochs, optimizer, criterion, train_loader, val_loader):
    current_directory = os.getcwd()
    save_folder_path = os.path.join(current_directory, 'SavedModels')
    create_folder(save_folder_path)

    train_loss = np.zeros(shape=epochs)
    train_acc  = np.zeros(shape=epochs)
    val_loss   = np.zeros(shape=epochs)
    val_acc    = np.zeros(shape=epochs)
    train_confusion_matrix = np.zeros(shape=(6,6,epochs))
    val_confusion_matrix   = np.zeros(shape=(6,6,epochs))

    train_predicted = [[] for _ in range(epochs)]
    train_true_vals = [[] for _ in range(epochs)]

    val_predicted = [[] for _ in range(epochs)]
    val_true_vals = [[] for _ in range(epochs)]

    for epoch in range(epochs):
        train_loss[epoch], train_acc[epoch], train_confusion_matrix[:,:,epoch], train_predicted[epoch], train_true_vals[epoch] = \
                                run_epoch(model, epoch, train_loader, optimizer, criterion, is_training=True)

        val_loss[epoch], val_acc[epoch], val_confusion_matrix[:,:,epoch], val_predicted[epoch], val_true_vals[epoch]     = \
                                run_epoch(model, epoch, val_loader, optimizer,criterion,  is_training=False)
        
        # Save the entire model after each epoch
        save_path = os.path.join(save_folder_path, f'model_checkpoint_epoch_{epoch}.pth')
        save_model(model, optimizer, epoch, save_path)
    
    return train_loss, train_acc, train_confusion_matrix, train_predicted, train_true_vals, val_loss, val_acc, val_confusion_matrix, val_predicted, val_true_vals

def plot_me(train_loss, train_acc, val_loss,val_acc, save_path):
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = plt.subplot(2, 1, 1)
    # plt.subplots_adjust(hspace=2)
    ax.plot(train_loss, 'b', label='train loss')
    ax.plot(val_loss, 'r', label='validation loss')
    ax.grid()
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    ax.legend(loc='upper right', fontsize=16)

    ax = plt.subplot(2, 1, 2)
    plt.subplots_adjust(hspace=0.4)
    ax.plot(train_acc, 'b', label='train accuracy')
    ax.plot(val_acc, 'r', label='validation accuracy')
    ax.grid()
    plt.ylabel('Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    val_acc_max = np.max(val_acc)
    val_acc_max_ind = np.argmax(val_acc)
    plt.axvline(x=val_acc_max_ind, color='g', linestyle='--', label='Highest validation accuracy')
    plt.title('Highest validation accuracy = %0.1f %%' % (val_acc_max*100), fontsize=16)
    ax.legend(loc='lower right', fontsize=16)

    plt.savefig(save_path)  
    plt.close()

def plot_precision_and_accuracy(maps, average_accuracy, save_path):
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax = plt.subplot(2, 1, 1)
    # plt.subplots_adjust(hspace=2)
    ax.plot(maps, 'b', label='mAP')
    ax.plot(average_accuracy, 'r', label='accuracy')
    ax.grid()
    plt.ylabel('Averages', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    ax.legend(loc='upper right', fontsize=16)

    plt.savefig(save_path)  
    plt.close()


############################### NEW code BABYYYY ###################################
def calc_accuracy_per_class(confusion_matrix, classes, epoch):
    accuracies = []

    for ii in range(len(classes)):
        acc = confusion_matrix[ii,ii,epoch] / np.sum(confusion_matrix[ii,:,epoch])
        print(f'Accuracy of {str(classes[ii]).ljust(15)}: {acc*100:.01f}%')
        accuracies.append(acc)
    
    average_over_classes = np.mean(accuracies)
    print("AVG:",average_over_classes)
    return average_over_classes, accuracies 

def accuracies_for_all_epochs(confusion_matrices, classes):
    epoch_accuracies = [[] for _ in range(confusion_matrices.shape[2])]
    averages = [0] * confusion_matrices.shape[2]

    for epoch in range(confusion_matrices.shape[2]):
        avg_accuracy, acc = calc_accuracy_per_class(confusion_matrices, classes, epoch)
        averages[epoch] = avg_accuracy
        epoch_accuracies[epoch] = acc 
        print()

    return averages, epoch_accuracies

'''
Calculates AP and mAP for a single epoch
'''
def calc_average_precision_per_class(y_true, y_pred, classes):
    average_precision = 0
    average_per_class = [0]*6
    for i in range(len(classes)):
        # Modify labels for binary classification
        y_true_bin = [1 if label == i else 0 for label in y_true]
        y_pred_bin = [1 if label == i else 0 for label in y_pred]
    
        precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        ap_i = average_precision_score(y_true_bin, y_pred_bin)
        average_per_class[i] = ap_i
        average_precision += ap_i
        #print(f'Class {i} - Average Precision: {ap_i}')
    
    average_precision /= 6

    #print(f'mAP: {average_precision}')
    return average_precision, average_per_class

def average_precisions_mAPs_for_all_epochs(y_true, y_pred, classes):
    epochs = len(y_true)
    mAPs = [0] * epochs
    epoch_precisions = [[] for _ in range(epochs)]
    for epoch in range(epochs):
        mAP, precisions = calc_average_precision_per_class(y_true[epoch], y_pred[epoch], classes)
        mAPs[epoch] = mAP
        epoch_precisions[epoch] = precisions
    
    return mAPs, epoch_precisions

def train_resnet18_model(device, num_epochs, train_loader, val_loader):
    # Define ANSI escape code for red color
    RED = '\033[91m'
    # Define ANSI escape code to reset color
    RESET = '\033[0m'
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = models.resnet50(pretrained=True)
    num_classes = 6 #len(dataset.class_to_idx)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)  # Move the model to GPU or CPU

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    train_loss, train_acc, train_confusion_matrix, train_predicted, train_true_vals, val_loss, val_acc, val_confusion_matrix, val_predicted, val_true_vals = run_the_training(model, num_epochs, optimizer, criterion, train_loader, val_loader)
    print("Training Losses")
    print(train_loss)
    print("------------\n")

    print("Validation Losses")
    print(val_loss)
    print("------------")

    plot_me(train_loss, train_acc,val_loss, val_acc,"Validation_loses.png")

    #accuracies for each class per epoch
    accuracies_averages, accuracies = accuracies_for_all_epochs(val_confusion_matrix, [0,1,2,3,4,5])
    
    print("Accuracies")
    print(accuracies_averages)
    #print(accuracies)
    print(f'{RED}Average over all epochs: {np.mean(accuracies_averages)*100:.01f}%{RESET}')

    #mAP and APs per class for each epoch
    mAPs, APs = average_precisions_mAPs_for_all_epochs(val_true_vals, val_predicted, [0,1,2,3,4,5])
    print("\nmAps")
    print(mAPs)
    {print(f'{RED}mAP over all epochs: {np.mean(mAPs)*100:.01f}%{RESET}')}
    plot_precision_and_accuracy(mAPs, accuracies_averages, "mAP_and_average_class_accuracy.png")


def create_datasets_and_loaders(root_path, transform=None, seed=50):

    def worker_init_fn(worker_id):
        torch.manual_seed(seed + worker_id)
    
    # Training dataset and dataloader
    train_dataset = ImageDataset(root_dir=root_path, train=True, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,worker_init_fn=worker_init_fn)

    # Validation dataset and dataloader
    val_dataset = ImageDataset(root_dir=root_path, train=False, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

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

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    num_epochs = 30 #30
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data sets and loaders
    train_loader, val_loader = create_datasets_and_loaders(root_path=root_path, transform=transform, seed=seed)

    #training and validation
    train_resnet18_model(device, num_epochs, train_loader, val_loader)