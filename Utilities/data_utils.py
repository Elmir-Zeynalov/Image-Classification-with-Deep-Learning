
import torch.utils.data as data
from PIL import Image
import torch
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms, models

class ImageDataset(data.Dataset):
    def __init__(self, root_dir, dataset="train", datapath="Datasets/", transform=None):
        self.root_dir = root_dir
        self.datapath = datapath
        self.transform = transform
        self.training_file = os.path.join(self.datapath,"train_set.txt") 
        self.validation_file = os.path.join(self.datapath,"validation_set.txt") 
        self.test_file = os.path.join(self.datapath,"test_set.txt") 
        self.dataset    = dataset
        self.class_to_idx = {}
        self.image_filenames = []
        self.labels = []
        self.transform = transform

        if self.dataset == "train":
            file_path = self.training_file
        elif self.dataset == "val":
            file_path = self.validation_file
        elif self.dataset == "test":
            file_path = self.test_file
        else:
            raise ValueError("Invalid value for 'train' parameter. Use 'train', 'val', or 'test'.")

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split(" ")
                self.image_filenames.append(parts[0])
                self.labels.append(parts[1].strip())

        self.class_to_idx = self.convert_label_to_ix(self.labels)
        print(f'SET: {self.dataset}')
        print(self.class_to_idx)
        print(f'Number of images in set: {len(self.labels)}\n')

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
        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return class_to_idx


class DataSplitter():
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
    def __init__(self, root_dir, dataset_path):
        self.root_dir = root_dir
        self.validation_size = 2000
        self.test_size = 3000
        self.dataset_path = dataset_path
        self.files, self.labels = self.load_files(self.root_dir)
        self.create_dataset(self.files, self.labels)
        print("Done creating train, validation and test sets...")

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

        self.save_file_paths(X_train_temp, y_train_temp, os.path.join(self.dataset_path,"train_set.txt"))
        self.save_file_paths(X_validation, y_validation, os.path.join(self.dataset_path,"validation_set.txt"))
        self.save_file_paths(X_test, y_test, os.path.join(self.dataset_path,"test_set.txt"))




def create_datasets_and_loaders(root_path, datasets_path, transform=None, seed=50):
    def worker_init_fn(worker_id):
        torch.manual_seed(seed + worker_id)
    
    # Training dataset and dataloader
    train_dataset = ImageDataset(root_dir=root_path, dataset="train", datapath=datasets_path,transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2,worker_init_fn=worker_init_fn)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Validation dataset and dataloader
    val_dataset = ImageDataset(root_dir=root_path, dataset="val",datapath=datasets_path, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    test_dataset = ImageDataset(root_dir=root_path, dataset="test",datapath=datasets_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    
    return train_loader, val_loader, test_loader, val_dataset