import pandas as pd
import pretty_errors
from data import (
    fetch_heart_disease_data, split_dataset, DataProcessor
)

def display_scaling_statistics(X, numerical_features):
    """
    Displays the mean and standard deviation for each numerical feature in the dataset.

    Parameters:
    - X: DataFrame, the dataset to analyze.
    - numerical_features: list of strings, the names of the numerical columns.
    """
    stats = pd.DataFrame(index=numerical_features, columns=['Mean', 'StdDev'])

    for feature in numerical_features:
        stats.loc[feature, 'Mean'] = X[feature].mean()
        stats.loc[feature, 'StdDev'] = X[feature].std()

    print(stats)

if __name__ == '__main__':

    # Fetch Features and Target
    dataframe = fetch_heart_disease_data()

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    processor = DataProcessor(categorical_features, numerical_features)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(dataframe)

    # Fit and transform the training data
    X_train_processed = processor.fit_transform(X_train)

    # Transform the validation and test data
    X_valid_processed = processor.transform(X_val)
    X_test_processed = processor.transform(X_test)
    print("Missing values in training data:", X_train_processed.isnull().sum().sum())
    print("Missing values in validation data:", X_valid_processed.isnull().sum().sum())
    print("Missing values in test data:", X_test_processed.isnull().sum().sum())



    print("Training Data:")
    display_scaling_statistics(X_train_processed, numerical_features)

    print("\nValidation Data:")
    display_scaling_statistics(X_valid_processed, numerical_features)

    print("\nTest Data:")
    display_scaling_statistics(X_test_processed, numerical_features)
    print("Encoded features in training data:", X_train_processed.columns)
    print("Encoded features in validation data:", X_valid_processed.columns)
    print("Encoded features in test data:", X_test_processed.columns)


