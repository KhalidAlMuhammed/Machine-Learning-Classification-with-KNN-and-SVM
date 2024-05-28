from mlmodel import (
    readCSV,
    datasetInfo,
    splitDataset,
    NNClassifier,
    SKLearnKnnClassifier,
    SKLearnSVMClassifier,
)


def main():
    df = readCSV(input("input a file name to load: "))

    # split dataset into training and testing data, with separate labels
    # make sure data is randomly shuffled beforehand!!
    train, train_labels, test, test_labels = splitDataset(df, test_percentage=20)

    # Print some info about test/train split
    print("Train dataset has {} entries".format(len(train)))
    print("Test dataset has {} entries".format(len(test)))

    # run our knn classifier
    k = int(input("How many nearest neighbors for handwritten KNN? "))
    print("Running knn classifier...")
    nn_accuracy = NNClassifier(train, test, train_labels, test_labels, k)

    # run the knn classifier from scikit-learn
    k = int(input("How many sklearn nearest neighbors? "))
    print("Running sklearn-knn classifier...")
    knn_accuracy = SKLearnKnnClassifier(train, test, train_labels, test_labels, k)

    # run the svm classifier from scikit-learn
    print("Running sklearn svm classifier...")
    svm_accuracy = SKLearnSVMClassifier(train, test, train_labels, test_labels)

    # print the accuracies
    print(
        "Accuracies:\n\tKNN (1006):{:.1%}\n\tKNN (sklearn):{:.1%}\n\tSVM (sklearn):{:.1%}".format(
            nn_accuracy, knn_accuracy, svm_accuracy
        )
    )


if __name__ == "__main__":
    main()
