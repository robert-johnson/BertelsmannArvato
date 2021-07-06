import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder


def impute_knn(df, neighbors=2, weights='uniform', verbose=False):
    """
    Imputes missing values using KNN algorithm
    :param df: dataframe to modify
    :param neighbors: number of neighbors to evaluate
    :param weights: weight strategy
    :param verbose: show progress
    :return: modified dataframe
    """
    if verbose:
        for c in df.columns:
            print(c, Counter(df[c]))

    i = KNNImputer(n_neighbors=neighbors, weights=weights, missing_values=-1)
    imputed = i.fit_transform(df)  # .to_numpy().reshape(-1, 1))

    df = pd.DataFrame(data=imputed, columns=[df.columns])

    if verbose:
        for c in df.columns:
            print(c, Counter(df[c]))

    return df


def impute_simple(df, columns=None, missing_values=np.nan, strategy='mean', verbose=False):
    """
    Imputes missing values using the SimpleImputer
    :param df: datafrome to modify
    :param columns: column names, if provided
    :param missing_values: missing values to impute
    :param strategy: impute strategy (mean, most_frequent)
    :param verbose: show progress
    :return: modified dataframe
    """
    imp = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputed = imp.fit_transform(df)  # .to_numpy().reshape(-1, 1)

    if columns is None:
        columns = df.columns

    df = pd.DataFrame(data=imputed, columns=columns)

    if verbose:
        print(Counter(df))

    return df


def encode_labels(df, col_name):
    """
    Encodes categorical data to numbers
    :param df: dataframe to modify
    :param col_name: column name to encode
    :return: modified dataframe
    """
    encoder = LabelEncoder()
    unique_labels = list(df[col_name].unique())

    encoder.fit(unique_labels)
    df[col_name] = encoder.transform(df[col_name])

    return df
