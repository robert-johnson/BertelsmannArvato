import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils.groups import subgroup_1, subgroup_2, subgroup_3, subgroup_4, drop_columns


def convert_toNaN(df, nan_value=np.nan):
    """
    Look for all the unknown or empty values and convert them to np.nan
    :param df: dataframe to convert
    :param nan_value: nan character to use
    :return: converted dataframe
    """

    # using the subgroup_1 to identify columns having the same missing/unknown indicator
    # missing values indicated by -1, 9
    df_subgroup_1 = df[subgroup_1]
    df_subgroup_1 = df_subgroup_1.replace([-1, 9], np.nan, inplace=False)
    df[subgroup_1] = df_subgroup_1[subgroup_1]

    # using the subgroup_2 to identify columns having the same missing/unknown indicator
    # missing values indicated by 10
    df_subgroup_2 = df[subgroup_2]
    df_subgroup_2 = df_subgroup_2.replace([10], np.nan, inplace=False)
    df[subgroup_2] = df_subgroup_2[subgroup_2]

    # using the subgroup_3 to identify columns having the same missing/unknown indicator
    # missing values indicated by -1, 0
    df_subgroup_3 = df[subgroup_3]
    df_subgroup_3 = df_subgroup_3.replace([-1, 0], np.nan, inplace=False)
    df[subgroup_3] = df_subgroup_3[subgroup_3]

    # using the subgroup_4 to identify columns having the same missing/unknown indicator
    # missing values indicated by -1, X, XX
    df_subgroup_4 = df[subgroup_4]
    df_subgroup_4 = df_subgroup_4.replace([-1, 'X', 'XX'], np.nan, inplace=False)
    df[subgroup_4] = df_subgroup_4[subgroup_4]

    # get all the other -1
    df.replace([-1], np.nan, inplace=True)

    # fill any other missing values
    df.fillna(np.nan, inplace=True)

    return df


def remove_columns(df, columns=None):
    """
    Removes the columns as defined in the columns param
    :param df: Dataframe to drop from
    :param columns: columns to drop, or None
    :return: updated dataframe
    """

    if columns is None:
        columns = drop_columns

    # let's drop the columns we've identified
    df_dropped = df.drop(columns=columns, inplace=False)

    return df_dropped


def drop_cols(df, threshold=0.5, append_cols=None):
    """
    Calculates the columns that have missing values gt threshold
    :param df: dataframe to process
    :param threshold: percentage of missing values
    :param append_cols: additional cols to add to the drop list
    :return: updated dataframe
    """

    cols = []
    # get the count of missing values
    missing = df.isnull().sum()
    # for each column in the df
    for c in df.columns:
        # calculate the missing values / number rows
        # if greater than threshold, append to drop list
        if (missing.loc[c] / df.shape[0]) > threshold:
            cols.append(c)

    # drop the columns in the list
    df_pruned = df.drop(cols, axis=1, inplace=False)

    # return the new dataframe
    return df_pruned


def drop_rows(df, threshold=0.5):
    """
    Drops the rows with missing values greater than threshold
    :param df: dataframe to process
    :param threshold: percent of missing values
    :return: updated dataframe
    """

    # calculate the count based on the threshold
    count = int(threshold * df.shape[1] + 1)
    # drop any rows that don't meet the count
    df_dropped = df.dropna(axis=0, thresh=count, inplace=False)

    # return the dropped dataframe
    return df_dropped


def get_components(cols, cumulative, threshold):
    """
    Helper method for PCA calculation
    :param cols: dataframe columns
    :param cumulative: cumulative variance
    :param threshold: threshhold
    :return: column index/component count
    """
    for i in range(cols):
        if cumulative[i] >= threshold:
            return i + 1


def get_pca(df, threshold=80, pc=None):
    """
    Calcualte the Principal Components
    :param df: dataframe to evaluate
    :param threshold: threshold
    :param pc: (optional) provided to use the same principal components for another dataframe
    :return: modified df, components, pca
    """
    scaled = scale(df)

    # if we're calculating, then no pc is passed
    if pc is None:
        # get the length of columns
        cols = len(list(df.columns))
        # new up the PCA with col length
        pca = PCA(n_components=cols)
        # fit the data
        pca.fit(scaled)

        # get the variance and the cumulative variance
        variance = pca.explained_variance_ratio_
        cumulative = np.cumsum(np.round(variance, decimals=3)*100)

        # determine how many features represent the threshold
        components = get_components(cols, cumulative, threshold)

    else:
        # if we do have a pc defined, use it
        components = pc

    # get the PCA for the components calculated
    pca = PCA(n_components=components, random_state=42)
    principal_components = pca.fit_transform(scaled)
    print(pca.n_components_)

    # create a new dataframe with the principal components
    df_pca = pd.DataFrame(data=principal_components, columns=['c_' + str(i) for i in range(1, components + 1)])
    # add the two datasets, then drop the existing columns
    df = pd.concat([df, df_pca], axis=1).drop(columns=list(df.columns))

    # return the dataframe, components and pca
    return df, components, pca


def scale(df):
    df.astype(dtype='float64')
    to_scale = df.copy()

    # scale the data
    scaled = StandardScaler().fit_transform(to_scale)

    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    print(df_scaled.head())

    return df_scaled


def oversample(X, y, strategy=0.5):

    oversampler = RandomOverSampler(sampling_strategy=strategy)
    # oversampler = SMOTE(sampling_strategy=strategy)

    X, y = oversampler.fit_resample(X, y)

    return X, y


def undersample(X, y, strategy=0.2):

    undersampler = RandomUnderSampler(sampling_strategy=strategy)

    X, y = undersampler.fit_resample(X, y)

    return X, y

