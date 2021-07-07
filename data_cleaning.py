from collections import Counter

import pandas as pd

from utils.impute import impute_simple, encode_labels
from utils.transform import get_pca, drop_cols, drop_rows, convert_toNaN, remove_columns
from utils.groups import drop_columns, drop_customer
from utils.cluster import calculate_kmeans, predict_kmeans, plot_kmeans, map_clusters, plot_clusters, get_feature_weights

if __name__ == '__main__':
    # first step is to sync up the columns across azdias, customer and the attributes
    # for me, if I don't have information on the columns, then I plan to drop them
    # we can also handle the mixed type error by specifying that the offending columns are str
    print('Loading data files')
    df_azdias = pd.read_csv('azdias.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')
    df_customers = pd.read_csv('customers.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')

    # convert to np.nan
    print('Converting missing/none/unknown to np.nan')
    df_azdias = convert_toNaN(df_azdias)
    df_customers = convert_toNaN(df_customers)
    # pd.set_option('display.max_rows', df_azdias.shape[1] + 1)

    print('Removing rows/cols that do not meet the minimum threshold')
    # remove the columns that don't have matched data
    df_azdias = remove_columns(df_azdias, drop_columns)
    drop_customer.extend(drop_columns)
    df_customers = remove_columns(df_customers, columns=drop_customer)

    # calculate which columns have less than threshold of data, drop them
    df_azdias = drop_cols(df_azdias)
    df_customers = drop_cols(df_customers)

    # calculate which rows have less than threshold values, drop them
    df_azdias = drop_rows(df_azdias, threshold=0.75)
    df_customers = drop_rows(df_customers, threshold=0.75)

    print('Updated shape:', df_azdias.shape)

    # now that we have removed the cols/rows, let's impute the missing data
    df_azdias = impute_simple(df_azdias, strategy='most_frequent')
    df_customers = impute_simple(df_customers, strategy='most_frequent')

    # let's encode the str labels to numbers
    df_azdias = encode_labels(df_azdias, 'CAMEO_DEU_2015')
    df_azdias = encode_labels(df_azdias, 'OST_WEST_KZ')

    df_customers = encode_labels(df_customers, 'CAMEO_DEU_2015')
    df_customers = encode_labels(df_customers, 'OST_WEST_KZ')

    # we still have too much data, let's look to figure out how to reduce the dimensionality
    df_azdias_pca, components, azdias_pca = get_pca(df_azdias, threshold=90)
    df_customers_pca, c_components, customers_pca = get_pca(df_customers, threshold=90, pc=components)
    print(df_azdias_pca.shape)

    # df_azdias_pca.to_csv('data/azdias_pca.csv', index=False)
    # df_customers_pca.to_csv('data/customers_pca.csv', index=False)

    kmeans, azdias_labels = calculate_kmeans(df_azdias_pca, clusters=12)
    azdias_map = map_clusters(azdias_labels, df_azdias_pca.shape[0])

    plot_kmeans(df_azdias_pca, 1, 13, 2)

    customers_labels = predict_kmeans(df_customers_pca, kmeans)
    customers_map = map_clusters(customers_labels, df_customers_pca.shape[0])

    plot_clusters(azdias_map, customers_map)

    for c in range(1, 13):
        azdias_weights = get_feature_weights(df=azdias_pca, pca=kmeans, cluster=c, columns=df_azdias.columns)
        pd.set_option('display.max_rows', df_azdias.shape[1] + 1)
        print(c)
        print(azdias_weights[0:5])
        print(azdias_weights[-5:])
        print('*******************************************************')
