import os
import torch
import shutil
import numpy as np

import torch.nn.functional as F
from .utils import load_presaved_model, create_folder
from sklearn import metrics
from .data_utils import ImageDataset
from torchvision import transforms, models
from .accuracy_precision_utils import accuracies_for_all_epochs, average_precisions_mAPs_for_all_epochs


def copy_images(image_paths, directory_name, classes=['buildings', 'forest', 'glacier']):
    '''
    This function creates 2 directories, Best_10 and Worst_10
    Under each directory thare are subdirectories representing 3 classes. 
    Images for the best and worst are then copied over to this local directory
    '''

    current_directory = os.getcwd()
    save_folder_path = os.path.join(current_directory, directory_name)
    create_folder(save_folder_path)

    for i, paths in enumerate(image_paths):
        class_path = os.path.join(save_folder_path, classes[i])
        create_folder(class_path)
        for img in paths:
            shutil.copy(img, class_path)

def find_best_and_worst_from_softmaxes(softmaxes, image_paths):
    best = [[] for _ in range(6)]
    worst = [[] for _ in range(6)]
 
    for class_id in range(6):
        class_scores = np.array([score[class_id] for score in softmaxes])
  
        top_indices = np.argsort(class_scores)[-10:][::-1].astype(int) 
        bottom_indices = np.argsort(class_scores)[:10].astype(int)

        best[class_id] = [image_paths[i] for i in top_indices]
        worst[class_id] = [image_paths[i] for i in bottom_indices]

    #choose the first 3 classes
    best = best[:3]
    worst = worst[:3]

    return best, worst

def evaluate_model_on_testset(model, dataloader, device, saveImages=False):
    '''
    F) Part 1 
    Predict on the test set, compute the mAP and mean accuracy per
    class, and save the softmax scores to file. For three classes of your choice, show
    ten images of the worst and ten of the best images according to the softmax
    score. 
    '''

    total_loss       = 0 
    correct          = 0 
    confusion_matrix = np.zeros(shape=(6,6))
    labels_list      = [0,1,2,3,4,5]

    predicted = []
    true_values = [] 
    softmax_scores = []
    image_paths = []

    model.eval()
    for batch_idx, data_batch in enumerate(dataloader):
        data = data_batch['image'].to(device)
        target = data_batch['label'].to(device)
        path = data_batch['filepath']
        
        # Forward pass through the model
        with torch.no_grad():
            output = model(data)
            softmax_output = F.softmax(output, dim=1)

        predicted_label  = output.max(1, keepdim=True)[1][:,0]
        confusion_matrix += metrics.confusion_matrix(target.cpu().numpy(), predicted_label.cpu().numpy(), labels=labels_list)
        predicted.extend(predicted_label.cpu().numpy())
        true_values.extend(target.cpu().numpy())
        softmax_scores.extend(softmax_output.cpu().numpy())
        image_paths.extend(path)

    confusion_matrix = confusion_matrix / len(dataloader.dataset)

    if saveImages:
        top_10, worst_10 = find_best_and_worst_from_softmaxes(softmax_scores, image_paths)
        copy_images(top_10, 'Best_10')
        copy_images(worst_10, 'Worst_10')

    return confusion_matrix, predicted, true_values, softmax_scores


def evaluate(model, test_loader, datasets_path, device, seed, num_epochs, saved_softmaxes_path, threshold):
    RED = '\033[91m'
    # Define ANSI escape code to reset color
    RESET = '\033[0m'
    confusion_matrix = np.zeros(shape=(6,6,1))
    predicted = [[] for _ in range(1)]
    true_values = [[] for _ in range(1)]
    #predicting and extracting softmaxes
    confusion_matrix[:,:, 0], predicted[0], true_values[0], softmax_scores = evaluate_model_on_testset(model, test_loader, device, saveImages=False)
    
    accuracies_averages, accuracies = accuracies_for_all_epochs(confusion_matrix, [0,1,2,3,4,5])
    mAPs, APs = average_precisions_mAPs_for_all_epochs(true_values, predicted, [0,1,2,3,4,5])
    
    print(f'\t\t[Evaluation]')
    print("***********************************************************")
    print("\t\tClass Accuracies")

    for i, c in enumerate(accuracies[0]):
        print(f'Accuracy of {str(i).ljust(15)}: {c*100:.01f}%')
    
    print(f'-> {RED}Avg Accuracy: {np.mean(accuracies_averages)*100:.01f}%{RESET}')
    print(f'-> {RED}AP: {APs}{RESET}')
    print(f'-> {RED}mAP: {np.mean(mAPs)*100:.01f}%{RESET}')
    print("***********************************************************")




def check_model_and_compare_with_saved_softmax(model_path, root_path, datasets_path, device, seed, num_epochs, saved_softmaxes_path, threshold):
    '''
    F) Part 2
    Write code to load the test set, predict on the test set, and then compare
    these against your saved softmax scores. There can be some tolerance between
    the two. Please use relative paths from the main Python files for loading the
    scores, model, etc. Only use an absolute path for the dataset root.
    '''

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #loading test set
    test_dataset = ImageDataset(root_dir=root_path, dataset="test", datapath=datasets_path ,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)


    #loading model
    model = load_presaved_model(device, seed, num_epochs-1, model_path)

    #predicting and extracting softmaxes
    _, _, _, softmax_scores = evaluate_model_on_testset(model, test_loader, device, saveImages=False)

    #loading saved softmaxes
    saved_softmax_scores = np.load(saved_softmaxes_path)

    #difference between softmaxes
    abs_diff = np.abs(saved_softmax_scores - softmax_scores)

    if np.any(abs_diff > threshold):
        print("The difference exceeds the threshold.\n")
    else:
        print("The difference is within the threshold.\n")


def test_model_and_extract_softmaxes(dataloader, device, num_epochs, softmax_path, seed):
    # Define ANSI escape code for red color
    RED = '\033[91m'
    # Define ANSI escape code to reset color
    RESET = '\033[0m'

    confusion_matrix = np.zeros(shape=(6,6,1))
    predicted = [[] for _ in range(1)]
    true_values = [[] for _ in range(1)]

    print(f'Running TEST set...')
    model = load_presaved_model(device, seed, num_epochs-1)
    confusion_matrix[:,:, 0], predicted[0], true_values[0], softmax_scores = evaluate_model_on_testset(model, dataloader, device,saveImages=True)
    np.save(softmax_path, np.array(softmax_scores))

    accuracies_averages, accuracies = accuracies_for_all_epochs(confusion_matrix, [0,1,2,3,4,5])
    mAPs, APs = average_precisions_mAPs_for_all_epochs(true_values, predicted, [0,1,2,3,4,5])

    print(f'\t\t[Evaluation]')
    print("***********************************************************")
    print("\t\tClass Accuracies")

    for i, c in enumerate(accuracies[0]):
        print(f'Accuracy of {str(i).ljust(15)}: {c*100:.01f}%')
    
    print(f'-> {RED}Avg Accuracy: {np.mean(accuracies_averages)*100:.01f}%{RESET}')
    print(f'-> {RED}mAP: {np.mean(mAPs)*100:.01f}%{RESET}')
    print("***********************************************************")


def test_model_and_compare_softmaxes(test_loader, device, num_epochs, seed, datasets_path, sotmax_path, threshold):
    test_model_and_extract_softmaxes(test_loader, device, num_epochs, os.path.join(sotmax_path,"softmax_scores.npy") ,  seed)
    check_model_and_compare_with_saved_softmax(model_path=f'SavedModels/model_checkpoint_epoch_{num_epochs-1}.pth', root_path='rr', datasets_path=datasets_path, device=device, seed=seed, num_epochs=num_epochs, saved_softmaxes_path=os.path.join(sotmax_path,"softmax_scores.npy"), threshold=threshold)
