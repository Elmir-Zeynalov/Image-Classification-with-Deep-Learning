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


def calc_class_accuracy(acc, confusion_matrix, classes):
    print("[Class Accuracies]")
    ind = np.argmax(acc)
    class_accuracy = confusion_matrix[:,:,ind]
    for ii in range(len(classes)):
        acc = confusion_matrix[ii,ii,ind] / np.sum(confusion_matrix[ii,:,ind])
        print(f'Accuracy of {str(classes[ii]).ljust(15)}: {acc*100:.01f}%')

def calc_class_tot_accuracy(acc, confusion_matrix, classes):
    print("[Class Accuracies]")
    ind = np.argmax(acc)
    class_accuracy = confusion_matrix[:,:,ind]
    accuracies = [[] for _ in range(len(classes))]

    for ii in range(len(classes)):
        for epoch in range(confusion_matrix.shape[2]):
            acc = confusion_matrix[ii,ii,epoch] / np.sum(confusion_matrix[ii,:,epoch])
            accuracies[ii].append(acc)
    
    data_array = np.array(accuracies)
    print("accur")
    print(accuracies)
    average_accuracies = np.mean(data_array, axis=1)

    print("[Average Accuracies]")
    for i, accuracy in enumerate(average_accuracies):
        print(f'Class {i}: {accuracy*100:.01f}%')
    return average_accuracies


def calc_class_precision(confusion_matrix, classes):
    print("[Class Precisions]")
    precisions = [[] for _ in range(len(classes))]

    for ii in range(len(classes)):
        for epoch in range(confusion_matrix.shape[2]):
            true_positives = confusion_matrix[ii, ii, epoch]
            false_positives = np.sum(confusion_matrix[:, ii, epoch]) - true_positives

            precision = true_positives / (true_positives + false_positives)
            precisions[ii].append(precision)

    avg_precisions = np.mean(precisions, axis=1)
    print(precisions)
    print("[Average Precisions]")
    for i, precision in enumerate(avg_precisions):
        print(f'Class {i}: {precision * 100:.01f}%')

    return avg_precisions


def calculate_precision(confusion_matrix):
    precision_values = []

    for epoch in range(confusion_matrix.shape[2]):
        epoch_matrix = confusion_matrix[:, :, epoch]

        for class_idx in range(epoch_matrix.shape[0]):
            true_positives = epoch_matrix[class_idx, class_idx]
            false_positives = np.sum(epoch_matrix[:, class_idx]) - true_positives

            if true_positives + false_positives == 0:
                precision = 0.0
            else:
                precision = true_positives / (true_positives + false_positives)

            precision_values.append(precision)

    return precision_values
    

def precisions(confusion_matrix_values):
    precision_values = calculate_precision(confusion_matrix_values)
    precisions = [[] for _ in range(6)]

    for epoch in range(confusion_matrix_values.shape[2]):
        print(f"Epoch {epoch + 1} Precision Values:")
        for class_idx in range(confusion_matrix_values.shape[0]):
            precisions[class_idx].append(precision_values[epoch * confusion_matrix_values.shape[0] + class_idx])
            print(f"Class {class_idx}: {precision_values[epoch * confusion_matrix_values.shape[0] + class_idx] * 100:.1f}%")
    

    data_array = np.array(precisions)

    average_precisions = np.mean(data_array, axis=1)
    print(average_precisions)
    print("\n[Average Precision]")
    for i, precision in enumerate(average_precisions):
        print(f'Class {i}: {precision*100:.01f}%')
    return average_precisions

def calculate_mAP(confusion_matrix_values):
    precision_values = calculate_precision(confusion_matrix_values)
    num_classes = confusion_matrix_values.shape[0]
    num_epochs = confusion_matrix_values.shape[2]

    ap_per_class = []

    for class_idx in range(num_classes):
        class_precisions = [precision_values[epoch * num_classes + class_idx] for epoch in range(num_epochs)]
        class_ap = np.mean(class_precisions)
        ap_per_class.append(class_ap)
        print(f'Class {class_idx} AP: {class_ap * 100:.1f}%')

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(ap_per_class)
    # Calculate mAP by averaging AP values across all classes
    mAP = np.mean(ap_per_class)
    print("\n[Mean Average Precision (mAP)]")
    print(f'mAP: {mAP * 100:.1f}%')

    return mAP


def calculate_ap(confusion_matrix_values):
    num_classes = confusion_matrix_values.shape[0]
    num_epochs = confusion_matrix_values.shape[2]
    ap_values = []

    for epoch in range(num_epochs):
        confusion_matrix = confusion_matrix_values[:, :, epoch]
        epoch_ap_values = []

        for class_idx in range(num_classes):
            true_positives = confusion_matrix[class_idx, class_idx]
            false_positives = np.sum(confusion_matrix[:, class_idx]) - true_positives
            false_negatives = np.sum(confusion_matrix[class_idx, :]) - true_positives

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0

            precision_values, recall_values, _ = precision_recall_curve([1 if i == class_idx else 0 for i in range(num_classes)],
                                                                        [1 if i == class_idx else 0 for i in range(num_classes)])

            # Find the index where recall is closest to a predefined set of values
            target_recalls = np.linspace(0, 1, num=100)  # adjust num based on your requirements
            closest_indices = np.argmin(np.abs(recall_values[:, None] - target_recalls), axis=0)

            # Debug prints
            print(f"Epoch {epoch + 1}, Class {class_idx}:")
            print(f"  Precision: {precision * 100:.1f}%")
            print(f"  Recall: {recall * 100:.1f}%")
            print(f"  AP from PR curve: {np.mean(precision_values[closest_indices]) * 100:.1f}%")

            # Calculate AP by taking the mean precision at the closest recall points
            ap = np.mean(precision_values[closest_indices])
            epoch_ap_values.append(ap)

        ap_values.append(epoch_ap_values)

    ap_values = np.array(ap_values)
    mAP_values = np.mean(ap_values, axis=0)

    print("\n[Mean Average Precision (mAP)]")
    for class_idx, mAP in enumerate(mAP_values):
        print(f'Class {class_idx} mAP: {mAP * 100:.1f}%')

    overall_mAP = np.mean(mAP_values)
    print(f'Overall mAP: {overall_mAP * 100:.1f}%')



def calculate_metrics(confusion_matrices):
    num_classes = confusion_matrices.shape[1]
    num_epochs = confusion_matrices.shape[2]

    accuracy_per_class = np.zeros((num_classes, num_epochs))
    precision_per_class = np.zeros((num_classes, num_epochs))
    average_precision_per_class = np.zeros((num_classes, num_epochs))

    for epoch in range(num_epochs):
        for class_label in range(num_classes):
            true_positives = confusion_matrices[class_label, class_label, epoch]
            false_positives = np.sum(confusion_matrices[class_label, :, epoch]) - true_positives
            false_negatives = np.sum(confusion_matrices[:, class_label, epoch]) - true_positives

            accuracy_per_class[class_label, epoch] = true_positives / np.sum(confusion_matrices[:, :, epoch])

            # Calculate precision and average precision
            if (true_positives + false_positives) > 0:
                precision_per_class[class_label, epoch] = true_positives / (true_positives + false_positives)
            else:
                precision_per_class[class_label, epoch] = 0.0

            labels = np.zeros_like(confusion_matrices[:, :, epoch])
            labels[class_label, :] = 1
            flat_labels = labels.flatten()
            flat_confusion = confusion_matrices[:, :, epoch].flatten()

            precision, recall, _ = precision_recall_curve(flat_labels, flat_confusion)
            average_precision_per_class[class_label, epoch] = average_precision_score(flat_labels, flat_confusion)

    # Calculate mean values over epochs
    accuracy_mean = np.mean(accuracy_per_class, axis=1)
    precision_mean = np.mean(precision_per_class, axis=1)
    average_precision_mean = np.mean(average_precision_per_class, axis=1)

    return accuracy_per_class, precision_per_class, average_precision_per_class, accuracy_mean, precision_mean, average_precision_mean



def calculate_accuracy_per_class(confusion_matrices):
    num_classes, _, num_epochs = confusion_matrices.shape
    accuracies_per_class = []

    for class_idx in range(num_classes):
        class_accuracies = []

        for epoch in range(num_epochs):
            true_positives = confusion_matrices[class_idx, class_idx, epoch]
            false_positives = np.sum(confusion_matrices[:, class_idx, epoch]) - true_positives
            false_negatives = np.sum(confusion_matrices[class_idx, :, epoch]) - true_positives

            accuracy = true_positives / (true_positives + false_positives + false_negatives)
            class_accuracies.append(accuracy)

        accuracies_per_class.append(class_accuracies)

    return np.array(accuracies_per_class)

def calculate_metr(confusion_matrix):
    # Calculate accuracy per class
    class_accuracy = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

    # Calculate precision per class
    class_precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)

    return class_accuracy, class_precision

def metrics_for_ecposg(confusion_matrix):
    print("calcing")
    for epoch in range(confusion_matrix.shape[2]):
        matrix = confusion_matrix[:, : , epoch]
        print(calculate_metr(matrix))
    print("done")

####################################################################################################### NEW code

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
    print(f'Average over all epochs: {np.mean(accuracies_averages)}')

    #mAP and APs per class for each epoch
    mAPs, APs = average_precisions_mAPs_for_all_epochs(val_true_vals, val_predicted, [0,1,2,3,4,5])
    print("\nmAps")
    print(mAPs)
    print(f'mAP over all epochs: {np.mean(mAPs)}')
    plot_precision_and_accuracy(mAPs, accuracies_averages, "mAP_and_average_class_accuracy.png")


#
#def train_resnet18_model(device, num_epochs, train_loader, val_loader):
#    torch.manual_seed(42)
#    torch.cuda.manual_seed_all(42)
#
#    model = models.resnet50(pretrained=True)
#    num_classes = 6 #len(dataset.class_to_idx)
#    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#
#    #dropout_prob = 0.2  # You can adjust this probability
#    #model.fc = torch.nn.Sequential(
#    #    torch.nn.Dropout(p=dropout_prob),  # Dropout layer
#    #    torch.nn.Linear(model.fc.in_features, num_classes)
#    #)
#    
#    model = model.to(device)  # Move the model to GPU or CPU
#
#    criterion = torch.nn.CrossEntropyLoss()
#    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
#
#
#    train_loss, train_acc, train_confusion_matrix, val_loss, val_acc, val_confusion_matrix = run_the_training(model, num_epochs, optimizer, criterion, train_loader, val_loader)
#    print("Training Losses")
#    print(train_loss)
#    print("------------")
#
#    print()
#    print("Validation Losses")
#    print(val_loss)
#    print("------------")
#
#
#    plot_me(train_loss, train_acc,val_loss, val_acc,"Validation_loses.png")
#    calc_class_accuracy(val_acc, val_confusion_matrix, [0,1,2,3,4,5])
#    print(val_confusion_matrix)
#    precisions(val_confusion_matrix)
#    print("GGG ---- GGG ---- GGGG ---- GGGG ----- GGGG ----GGG ----")
#    calculate_ap(val_confusion_matrix)
#    print("GGG ---- GGG ---- GGGG ---- GGGG ----- GGGG ----GGG ----\n\n")
#    calc_class_tot_accuracy(val_acc, val_confusion_matrix, [0,1,2,3,4,5])
#    calculate_mAP(val_confusion_matrix)
#
#
#    print("CALCULATIONS")
#    accuracy_per_class, precision_per_class, average_precision_per_class, accuracy_mean, precision_mean, average_precision_mean = calculate_metrics(val_confusion_matrix)
#    print(accuracy_per_class)
#    print("----  ----  ----  -----  ---- ----")
#
#    print(precision_per_class)
#    print("----  ----  ----  -----  ---- ----")
#    print(average_precision_per_class)
#    print("----  ----  ----  -----  ---- ----")
#    print(accuracy_mean)
#    print("----  ----  ----  -----  ---- ----")
#    print(precision_mean)
#    print("----  ----  ----  -----  ---- ----")
#    print(average_precision_mean)
#
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
    
    num_epochs = 5 #30
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data sets and loaders
    train_loader, val_loader = create_datasets_and_loaders(root_path=root_path, transform=transform, seed=seed)

    #training and validation
    train_resnet18_model(device, num_epochs, train_loader, val_loader)