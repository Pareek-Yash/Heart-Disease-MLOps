from ucimlrepo import fetch_ucirepo
import pandas as pd 

def fetch_heart_disease_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the Heart Disease dataset from the UCI Machine Learning Repository.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames Features and Labels.
    """
    # Fetch the Heart Disease dataset
    heart_disease_dataset = fetch_ucirepo(id=45)

    # Extract features and targets into separate DataFrames
    features = pd.DataFrame(heart_disease_dataset.data.features)
    target = pd.DataFrame(heart_disease_dataset.data.targets)

    dataframe = pd.concat([features, target], axis=1)

    dataframe.rename(columns={'num': 'target'}, inplace=True)

    return dataframe