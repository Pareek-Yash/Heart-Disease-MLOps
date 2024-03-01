import pandas as pd
from data import fetch_heart_disease_data
from data import split_dataset

if __name__ == '__main__':

    # Fetch Features and Target
    dataframe = fetch_heart_disease_data()
    print(dataframe.head())
    
    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(dataframe)
    # print('\nDataFrame split successfully\n')

    # print(f'X_train: \n{X_train.head()}')
    # print(f'y_train: \n{y_train.head()}')