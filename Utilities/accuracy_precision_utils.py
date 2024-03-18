import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, average_precision_score



def calc_accuracy_per_class(confusion_matrix, classes, epoch):
    accuracies = []

    for ii in range(len(classes)):
        acc = confusion_matrix[ii,ii,epoch] / np.sum(confusion_matrix[ii,:,epoch])
        #print(f'Accuracy of {str(classes[ii]).ljust(15)}: {acc*100:.01f}%')
        accuracies.append(acc)
    
    average_over_classes = np.mean(accuracies)
    #print("AVG:",average_over_classes)
    return average_over_classes, accuracies 

def accuracies_for_all_epochs(confusion_matrices, classes):
    epoch_accuracies = [[] for _ in range(confusion_matrices.shape[2])]
    averages = [0] * confusion_matrices.shape[2]

    for epoch in range(confusion_matrices.shape[2]):
        avg_accuracy, acc = calc_accuracy_per_class(confusion_matrices, classes, epoch)
        averages[epoch] = avg_accuracy
        epoch_accuracies[epoch] = acc 

    return averages, epoch_accuracies


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