import pandas as pd
import torch
import numpy as np

class Process:

    def impute():
        pass

    def encode():
        pass

    def sampling():
        pass

    def transform():
        pass


def process_heart_dataset(dataframe: pd.DataFrame):
    p = Process()
    transformed_features, transformed_label = p.impute()
    return

def split_dataset(dataframe: pd.DataFrame):
    """
    Splits features and labels into training, validation, and test sets.
    
    Parameters:
    dataframe (pd.DataFrame): The feature data.
    
    Returns:
    tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    
    
    # Shuffle the dataset: train 60%, validation 20%, test 20%
    train, val, test = np.split(dataframe.sample(frac=1, random_state=42), [int(0.6*len(dataframe)), int(0.8*len(dataframe))])

    # Separate the features and labels again
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
