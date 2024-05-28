import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatterMatrix(df, count=5):
    """Use seaborn to produce a pairplot of columns

    count: number of columns to scatter (larger will result in slower)
    """
    columns = df.columns[:count+1]  # Select the first count+1 columns (including the target variable)
    sns.pairplot(df[columns], hue=columns[0]) 
    plt.show()


def correlationHeatmap(df):
    """Use seaborn to produce a heatmap of the columns' correlation"""
    # Select only the numeric columns
    numeric_columns = [col for col in df.columns if df[col].dtype.kind in 'ifc']
    # Create a new dataframe with only the numeric columns
    numeric_df = df[numeric_columns]
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    # Create the heatmap
    sns.heatmap(corr_matrix)
    plt.show()
