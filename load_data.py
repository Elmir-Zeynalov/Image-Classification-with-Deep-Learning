import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from sklearn import metrics
import torch.nn.functional as F

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



def train(model, optimizer, criterion, num_epochs, data_loader):
    print("---Training---")
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs = batch['image']
            labels = batch['label']

            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def validate(model, criterion, num_epochs, data_loader):
    correct = 0
    total_samples = 0
    confusion_matrix = np.zeros(shape=(10,10))
    #accuracies    = np.zeros(shape=num_epochs)
    accuracies = []
    labels_list= [0,1,2,3,4,5]

    confusion_matrix = np.zeros((6, 6))
    class_accuracies = np.zeros(6)
    class_precisions = np.zeros(6)
    class_recalls = np.zeros(6)
    class_f1_scores = np.zeros(6)
    class_aps = np.zeros(6)


    print("---Validating---")
    model.eval()
    with torch.no_grad():
        for epoch in range(num_epochs):
            correct = 0
            total_samples = 0
            epoch_labels = []
            epoch_predictions = []
            for batch in data_loader:
                inputs = batch['image']
                labels = batch['label']
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                predicted_label = outputs.argmax(dim=1)
                correct += (predicted_label == labels).sum().item()
                total_samples += labels.size(0)

                confusion_matrix += metrics.confusion_matrix(labels.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)
                # Append labels and predictions for calculating AP later
                epoch_labels.extend(labels.cpu().numpy())
                epoch_predictions.extend(F.softmax(outputs, dim=1).cpu().numpy())

            accuracy = correct/total_samples
            accuracies.append(accuracy)
            for class_index in range(6):
                class_labels = [1 if label == class_index else 0 for label in epoch_labels]
                class_predictions = [prediction[class_index] for i, prediction in enumerate(epoch_predictions)]
                
                #class_accuracies[class_index] = metrics.accuracy_score(class_labels, np.array(class_predictions).round())
                class_accuracies[class_index] = metrics.accuracy_score(class_labels, np.array(class_predictions).round(), normalize=False)
                class_precisions[class_index] = metrics.precision_score(class_labels, np.array(class_predictions).round())
                class_recalls[class_index] = metrics.recall_score(class_labels, np.array(class_predictions).round())
                class_f1_scores[class_index] = metrics.f1_score(class_labels, np.array(class_predictions).round())
                class_aps[class_index] = metrics.average_precision_score(class_labels, np.array(class_predictions))

            #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            # Print metrics for each class at the end of the epoch
            print(f'Epoch {epoch + 1}: Accuracy: {accuracy * 100:.2f}%')
            for class_index in range(6):
                print(f'  Class {class_index}: Accuracy: {class_accuracies[class_index] / total_samples * 100:.2f}%, Precision: {class_precisions[class_index]:.4f}, Recall: {class_recalls[class_index]:.4f}, F1 Score: {class_f1_scores[class_index]:.4f}, AP: {class_aps[class_index]:.4f}')
                #print(f'  Class {class_index}: Accuracy: {class_accuracies[class_index] * 100:.2f}%, Precision: {class_precisions[class_index]:.4f}, Recall: {class_recalls[class_index]:.4f}, F1 Score: {class_f1_scores[class_index]:.4f}, AP: {class_aps[class_index]:.4f}')

    #accuracy = correct / len(data_loader.dataset)
    #accuracy = correct / total_samples
    #confusion_matrix = confusion_matrix / total_samples
    mAP = np.mean(class_aps)

    #print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    #print("\nAccuracies for each epoch:")
    #for epoch, acc in enumerate(accuracies):
    #    print(f'Epoch {epoch + 1}: Accuracy: {acc * 100:.2f}%')
    
    print("\nAccuracies for each epoch:")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}: Accuracy: {accuracies[epoch] * 100:.2f}%')

    print(f'Mean Average Precision (mAP): {mAP:.4f}')
    print("Confusion Matrix:")
    print(confusion_matrix)

    return accuracy

def train_resnet18_model(device, train_loader, val_loader):
    model = models.resnet18(pretrained=True)
    num_classes = 6 #len(dataset.class_to_idx)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)  # Move the model to GPU or CPU

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train(model, optimizer, criterion, num_epochs, train_loader)
    print()
    validate(model, criterion, num_epochs, val_loader)


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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data sets and loaders
    train_loader, val_loader = create_datasets_and_loaders(root_path=root_path, transform=transform, seed=seed)

    #training and validation
    train_resnet18_model(device, train_loader, val_loader)