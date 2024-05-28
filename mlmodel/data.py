from numpy.random import shuffle


def datasetInfo(dataset):
    """
    Takes the dataset and returns the following statistics as a dictionary:
        rows: number of rows in the dataset
        columns: number of columns in the dataset
        benign: Number of benign entries in dataset
        malignant: Number of malignant entries in dataset
    """
    # Get the number of rows in the dataset
    num_rows = len(dataset)
    
    # Get the number of columns in the dataset
    num_columns = len(dataset.columns)
    
    # Count the number of benign and malignant entries in the dataset
    diagnosis_counts = dataset.iloc[:, 0].value_counts()
    num_benign = diagnosis_counts['B'] if 'B' in diagnosis_counts else 0
    num_malignant = diagnosis_counts['M'] if 'M' in diagnosis_counts else 0
    
    # Create a dictionary with the dataset statistics
    dataset_info = {
        'rows': num_rows,
        'columns': num_columns,
        'benign': num_benign,
        'malignant': num_malignant
    }
    
    return dataset_info



def splitDataset(dataset, test_percentage=20):
    """
    Takes the dataset as a dataframe

    Shuffles data, then returns 4 subsets of the dataset as numpy matrices:
        Training data without labels (100-test_percentage percent of the data)
        Training data labels (column vector!)
        Testing data without labels (test_percentage percent of the data)
        Testing data labels (column vector!)
    """
        # Shuffle the dataset
    shuffled_df = dataset.sample(frac=1).reset_index(drop=True)
    
    # Split the dataset into training and testing sets
    test_size = int(len(shuffled_df) * test_percentage / 100)
    train_df = shuffled_df[:-test_size]
    test_df = shuffled_df[-test_size:]
    
    # Extract the labels from the first column
    train_labels = train_df.iloc[:, 0].values
    test_labels = test_df.iloc[:, 0].values
    
    # Remove the labels from the dataframes
    train_df = train_df.iloc[:, 1:]
    test_df = test_df.iloc[:, 1:]
    
    return train_df.values, train_labels, test_df.values, test_labels

