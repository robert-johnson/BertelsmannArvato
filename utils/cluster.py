import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def plot_kmeans(df, k_start=1, k_end=10, step=2):
    """
    Plots the clusters of the dataframe
    :param df: dataframe to plot
    :param k_start: starting cluster
    :param k_end: ending cluster
    :param step: plot every step
    :return: nothing
    """
    scores = []
    # loop through the range start/end
    for i in range(k_start, k_end, step):
        # new up a kmeans, with clusters
        kmeans = KMeans(n_clusters=i)
        km = kmeans.fit(df)
        # append the scores
        scores.append(abs(km.score(df)))

    # create a plot
    plt.figure(figsize=(14, 7))
    plt.plot(range(k_start, k_end, step), scores, marker='h', color='r')
    plt.xlabel('Cluster Count')
    plt.ylabel('Errors')
    plt.title('Errors by Cluster')
    plt.show()
    # plt.savefig('cluster_plot.png')


def calculate_kmeans(df, clusters=10):
    """
    Calculate the KMeans for the number of specified clusters
    :param df: dataframe to evaluate
    :param clusters: number of clusters
    :return: kmeans, labels
    """
    kmeans = KMeans(n_clusters=clusters)
    labels = kmeans.fit_predict(df)

    return kmeans, labels


def predict_kmeans(df, kmeans):
    """
    Using the passed kmeans, predict the labels for the dataframe
    :param df: dataframe to predict
    :param kmeans: kmeans to use
    :return: labels
    """
    labels = kmeans.predict(df)

    return labels


def map_clusters(labels, rows):
    """
    Maps the clusters to columns
    :param labels: cluster labels
    :param rows: rows
    :return: mappings
    """
    counts = Counter(labels)
    mappings = {c + 1: ((counts[c] / rows) * 100) for c in sorted(counts)}

    return mappings


def plot_clusters(cluster_1, cluster_2):
    """
    Plot the clusters side-by-side
    :param cluster_1: cluster 1
    :param cluster_2: cluster 2
    :return: Nothing
    """
    plt.figure(figsize=(14, 7))
    plt.bar([i - 0.1 for i in cluster_1.keys()], cluster_1.values(), width=0.2, align='center', color='b',
            label='German Population')
    plt.bar([i + 0.1 for i in cluster_2.keys()], cluster_2.values(), width=0.2, align='center', color='g',
            label='Customer Population')
    plt.title('German Population versus Customers')
    plt.xlabel('Cluster No.')
    plt.ylabel('Cluster %')
    plt.xticks(range(1, len(cluster_1) + 1))
    plt.legend()
    plt.savefig('cluster_map.png')
    plt.show()

    return


def get_feature_weights(df, pca, cluster, columns):
    """
    Gets the name of the features and their weights
    :param df: pca dataframe
    :param pca: pca model
    :param cluster: which cluster to evaluate
    :param columns: column map
    :return: sorted weigths
    """
    feature_weights = df.inverse_transform(pca.cluster_centers_[cluster-1])

    return pd.Series(feature_weights, index=columns).sort_values()
