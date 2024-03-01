import pandas as pd
from ydata_profiling import ProfileReport


def eda(dataframe):
    """
    Make yprofile eda.html file for Exploratory Data Analysis
    """
    profile = ProfileReport(dataframe, title='Heart Disease Dataset Report')
    profile.to_file('visualization/eda.html')


if __name__ == '__main__':
    eda()
    