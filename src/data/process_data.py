import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

class DataProcessor:
    def __init__(self, categorical_features, numerical_features, num_impute_strategy='mean', cat_impute_strategy='most_frequent'):
        """
        Initializes the DataProcessor with separate imputers for numerical and categorical features,
        an encoder for categorical features, and a scaler for numerical features.

        Parameters:
        - categorical_features: list of strings, the names of the categorical columns.
        - numerical_features: list of strings, the names of the numerical columns.
        - num_impute_strategy: string, the strategy for imputing numerical features ('mean', 'median', 'most_frequent', or 'constant').
        - cat_impute_strategy: string, the strategy for imputing categorical features ('most_frequent' or 'constant').
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.num_imputer = SimpleImputer(strategy=num_impute_strategy)
        self.cat_imputer = SimpleImputer(strategy=cat_impute_strategy, fill_value='missing')
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fits the imputers using the provided features from the training data.

        Parameters:
        - X: DataFrame, the training data.
        """

        if self.numerical_features:
            # Impute the numerical features
            self.num_imputer.fit(X[self.numerical_features])
            X_num_imputed = pd.DataFrame(self.num_imputer.transform(X[self.numerical_features]), columns=self.numerical_features)
            # Fit the scaler on the imputed data
            self.scaler.fit(X_num_imputed.values)

        if self.categorical_features:
            self.cat_imputer.fit(X[self.categorical_features])
            X_cat_imputed = pd.DataFrame(self.cat_imputer.transform(X[self.categorical_features]), columns=self.categorical_features)
            self.encoder.fit(X_cat_imputed)
            

        return self

    def transform(self, X):
        """
        Applies the imputations to the given dataset.

        Parameters:
        - X: DataFrame, the data to transform.

        Returns:
        - X_transformed: DataFrame, the transformed data.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        X_transformed = X.copy()

        if self.numerical_features:
            X_num = X[self.numerical_features]
            # Impute and scale the numerical features
            X_num_transformed = self.num_imputer.transform(X_num)
            X_num_scaled = self.scaler.transform(X_num_transformed)
            X_transformed[self.numerical_features] = X_num_scaled

        if self.categorical_features:
            # Apply imputation to the categorical features
            X_cat_imputed = self.cat_imputer.transform(X[self.categorical_features])

            # Apply the encoder to the imputed data
            X_cat_encoded = self.encoder.transform(X_cat_imputed)

            # Drop the original categorical columns
            X_transformed.drop(self.categorical_features, inplace=True, axis=1)

            # Create a DataFrame from the encoded features and concatenate it with the existing DataFrame
            encoded_df = pd.DataFrame(X_cat_encoded.toarray(), columns=self.encoder.get_feature_names_out(), index=X.index)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)


        return X_transformed

    def fit_transform(self, X):
        """
        Fits the imputer and transforms the dataset in one step.

        Parameters:
        - X: DataFrame, the training data to fit and transform.

        Returns:
        - X_transformed: DataFrame, the transformed data.
        """
        return self.fit(X).transform(X)



def split_dataset(dataframe: pd.DataFrame):
    """
    Splits features and labels into training, validation, and test sets.

    Parameters:
    dataframe (pd.DataFrame): The feature data.

    Returns:
    tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """

    # Shuffle the dataset: train 60%, validation 20%, test 20%
    shuffled_df = dataframe.sample(frac=1, random_state=42)
    train_end = int(0.6 * len(shuffled_df))
    val_end = int(0.8 * len(shuffled_df))

    train = shuffled_df.iloc[:train_end]
    val = shuffled_df.iloc[train_end:val_end]
    test = shuffled_df.iloc[val_end:]

    # Separate the features and labels again
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_val, y_val = val.iloc[:, :-1], val.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
