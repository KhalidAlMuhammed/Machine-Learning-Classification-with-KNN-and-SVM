import numpy as np
from math import sqrt
from statistics import mode


def NNClassifier(training, testing, training_labels, testing_labels, k):
    # preallocate labels
    predicted_labels = [""] * len(testing_labels)

    # run knn on each point and assign its label
        # fill in the labels with knn
    for i in range(len(testing)):
        predicted_labels[i] = knn(training, training_labels, testing[i], k)
    
    
    # return % where prediction matched actual
    accuracy = sum(label1 == label2 for label1, label2 in zip(predicted_labels, testing_labels)) / len(testing_labels)
    return accuracy

def knn(data, data_labels, vector, k):
    # preallocate distance array
    distances = np.zeros(len(data_labels))
    # calculate distances
    for i in range(len(data)):
        distance = 0
        for j in range(len(vector)):
            distance += (vector[j] - data[i][j]) ** 2
        distances[i] = distance ** 0.5
    # set labels
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    k_indices = sorted_indices[:k]
    k_labels = [data_labels[i] for i in k_indices]

    # take vote amongs top labels
    predicted_label = mode(k_labels)[0][0]
    return predicted_label

