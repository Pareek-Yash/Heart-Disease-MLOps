# Local imports
from data.make_dataset import fetch_heart_disease_data
from visualization.eda import eda

if __name__ == '__main__':
    dataframe = fetch_heart_disease_data()
    eda(dataframe)