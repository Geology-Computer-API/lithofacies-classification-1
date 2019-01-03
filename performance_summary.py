'''
Credit:
The code of the function below is a modified version of an example code in Sklearn
URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
'''

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np





def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cmap = plt.cm.Blues

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    #plt.figure()
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")

    print(cnf_matrix)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_accuracy_within_one_facies(labels, predictions):
    correct = 0
    for i in range(len(labels)):
        label = labels[i]
        prediction = predictions[i]
        if label == 1:
            if prediction in (1, 2):
                correct += 1
        elif label == 2:
            if prediction in (1, 2, 3):
                correct += 1
        elif label == 3:
            if prediction in (2, 3):
                correct += 1
        elif label == 4:
            if prediction in (4, 5):
                correct += 1
        elif label == 5:
            if prediction in (4, 5, 6):
                correct += 1
        elif label == 6:
            if prediction in (5, 6, 7):
                correct += 1
        elif label == 7:
            if prediction in (6, 7, 8):
                correct += 1
        elif label == 8:
            if prediction in (6, 7, 8, 9):
                correct += 1
        elif label == 9:
            if prediction in (7, 8, 9):
                correct += 1

    correct = float(correct)

    return correct / len(labels)
