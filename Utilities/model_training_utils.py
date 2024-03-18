import os
import torch
import numpy as np
from sklearn import metrics
from .plot_utils import plot_losses_and_accuracies, plot_precision_and_accuracy
from .accuracy_precision_utils import average_precisions_mAPs_for_all_epochs, accuracies_for_all_epochs
from .utils import create_resnet50_model, create_folder, save_model

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
                #print("EPOCH:", epoch , "batchid:", batch_idx, "loss:", loss)    
            
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
        print(f'Epoch = {epoch}')
        train_loss[epoch], train_acc[epoch], train_confusion_matrix[:,:,epoch], train_predicted[epoch], train_true_vals[epoch] = \
                                run_epoch(model, epoch, train_loader, optimizer, criterion, is_training=True)

        val_loss[epoch], val_acc[epoch], val_confusion_matrix[:,:,epoch], val_predicted[epoch], val_true_vals[epoch]     = \
                                run_epoch(model, epoch, val_loader, optimizer,criterion,  is_training=False)
        
        
    save_path = os.path.join(save_folder_path, f'model_checkpoint_epoch_{epoch}.pth')
    save_model(model, save_path)
    
    return train_loss, train_acc, train_confusion_matrix, train_predicted, train_true_vals, val_loss, val_acc, val_confusion_matrix, val_predicted, val_true_vals, model

def train_resnet_model(device, num_epochs, graphs_path, train_loader, val_loader, seed):
    # Define ANSI escape code for red color
    RED = '\033[91m'
    # Define ANSI escape code to reset color
    RESET = '\033[0m'
    classes = [0,1,2,3,4,5]

    model, optimizer, criterion = create_resnet50_model(device, seed)
    train_loss, train_acc, train_confusion_matrix, train_predicted, train_true_vals, val_loss, val_acc, val_confusion_matrix, val_predicted, val_true_vals, modello = run_the_training(model, num_epochs, optimizer, criterion, train_loader, val_loader)
    print("Training Losses")
    print(train_loss)
    print("------------\n")

    print("Validation Losses")
    print(val_loss)
    print("------------")

    plot_losses_and_accuracies(train_loss, train_acc,val_loss, val_acc, os.path.join(graphs_path,"Validation_loses.png"))

    #accuracies for each class per epoch
    accuracies_averages, accuracies = accuracies_for_all_epochs(val_confusion_matrix, classes)
    
    print(f'\nValidation Accuracies for last epoch {num_epochs}')
    #print(accuracies_averages)
    for i, c in enumerate(accuracies[num_epochs-1]):
        print(f'Accuracy of {str(i).ljust(15)}: {c*100:.01f}%')

    # Uncomment the code below if you want to see the class accuracy for every epoch.
    #print(f'\nValidation Accuracies for all epochs (per epoch)')
    #for epoch, acc in enumerate(accuracies):
    #    print(f'Epoch: {epoch}')
    #    for ii, accuracy in enumerate(acc):
    #        print(f'Accuracy of {str(ii).ljust(15)}: {accuracy*100:.01f}%')
    #    print()

    print(f'{RED}Average over all epochs: {np.mean(accuracies_averages)*100:.01f}%{RESET}')

    #mAP and APs per class for each epoch
    mAPs, APs = average_precisions_mAPs_for_all_epochs(val_true_vals, val_predicted, classes)
    print("\nValidation mAps")
    print(mAPs)
    print(f'{RED}mAP over all epochs: {np.mean(mAPs)*100:.01f}%{RESET}')
    plot_precision_and_accuracy(mAPs, accuracies_averages, os.path.join(graphs_path,"class_mAP_and_Acuracy.png"))  
    print("******Training complete******\n")
    return modello
