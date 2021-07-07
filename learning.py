import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from utils.transform import convert_toNaN, oversample, undersample, scale
from utils.impute import impute_simple, encode_labels
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

def train():
    df_train = pd.read_csv('mailout_train.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')
    # df_azdias = pd.read_csv('data/azdias_corrected.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'},
    #                         sep=',')
    pd.set_option('display.max_rows', df_train.shape[1] + 1)

    # print(sorted(Counter(df_train)))
    # print(sorted(Counter(df_azdias)))
    print(df_train.describe(exclude=np.number))
    df_train.drop([
        'Unnamed: 0',
        'D19_LETZTER_KAUF_BRANCHE',
        'EINGEFUEGT_AM',
        # 'OST_WEST_KZ',
        'LNR',
        'ALTER_KIND1',
        'ALTER_KIND2',
        'ALTER_KIND3',
        'ALTER_KIND4'
    ], axis=1, inplace=True)

    print(df_train.columns)
    # convert the the missing values to np.nan
    df_train = convert_toNaN(df_train)
    # impute the missing values
    # df_train['CAMEO_DEU_2015'] = impute_simple(df_train['CAMEO_DEU_2015'].to_numpy().reshape(-1, 1), columns=['CAMEO_DEU_2015'], strategy='most_frequent')
    # df_train = encode_labels(df_train, 'CAMEO_DEU_2015')
    df_train = impute_simple(df_train, strategy='most_frequent')
    # convert the categorical to numbers
    df_train = encode_labels(df_train, 'CAMEO_DEU_2015')
    df_train = encode_labels(df_train, 'OST_WEST_KZ')
    # df_train = encode_labels(df_train, 'D19_LETZTER_KAUF_BRANCHE')

    # df_train['EINGEZOGENAM_HH_JAHR'] = 0.01 * df_train['EINGEZOGENAM_HH_JAHR']
    # df_train['GEBURTSJAHR'] = 0.01 * df_train['GEBURTSJAHR']
    # df_train['KBA13_ANZAHL_PKW'] = 0.01 * df_train['KBA13_ANZAHL_PKW']
    # df_train['MIN_GEBAEUDEJAHR'] = 0.01 * df_train['MIN_GEBAEUDEJAHR']

    # print(Counter(df_train['RESPONSE']))

    # cols = list(np.setdiff1d(np.array(df_train.columns), np.array(df_azdias.columns)))
    # cols.remove('RESPONSE')
    # print(cols)
    # df_train.drop(cols, axis=1, inplace=True)
    # df_train = scale(df_train)
    # print(sorted(Counter(df_train)))

    X = df_train.drop('RESPONSE', axis=1)
    # X = scale(X)
    y = df_train['RESPONSE'].copy().astype('int')

    print(Counter(y))

    # X, y = oversample(X, y, strategy=0.2)
    # X, y = undersample(X, y, strategy=0.4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train, y_train = oversample(X_train, y_train, strategy=0.4)

    print(df_train.shape)
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    print(Counter(y_train))

    # df_train = convert_toNaN(df_train)
    # df_train = impute_simple(df_train, strategy='most_frequent')
    # print(df_train.isnull().sum())
    #
    # print(df_train.shape)
    # print(df_azdias.shape)
    # cols = np.setdiff1d(np.array(df_train.columns), np.array(df_azdias.columns))
    # df_train.drop(cols, axis=1, inplace=True)
    # print(df_train.shape)
    #
    # df_train.fillna(np.nan)
    # print(df_train.isnull().sum())

    scaler = StandardScaler()
    over = RandomOverSampler(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)
    cls = GradientBoostingClassifier(n_estimators=100, min_samples_split=4, max_depth=3, learning_rate=0.1, random_state=42)

    # cls = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # cls = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


    steps = [
        ('scaler', scaler),
        ('over', over),
        ('under', under),
        ('cls', cls)
    ]

    pipeline = Pipeline(steps=steps)

    pipeline.fit(X_train, y_train)

    y_preds = pipeline.predict_proba(X_test)

    score = roc_auc_score(y_test, y_preds[:, 1])
    print(score)

    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    # scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('Score: ', scores.mean())
    # print(pipeline.get_params())
    #
    # params = {
    #     'over__sampling_strategy': [0.1, 0.2],
    #     'under__sampling_strategy': [0.5, 0.6],
    #     'cls__n_estimators': [100],
    #     'cls__learning_rate': [0.1, 0.2],
    #     'cls__max_depth': [3, 5],
    #     'cls__min_samples_split': [2, 4]
    # }
    # # }
    # #
    # cv = GridSearchCV(pipeline, param_grid=params, scoring='roc_auc', n_jobs=-1)
    # cv.fit(X_train, y_train)
    #
    # print(cv.best_params_)
    # print(cv.best_score_)

    return pipeline


def test(pipeline):
    df_test = pd.read_csv('mailout_test.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')

    df_test.drop([
        'Unnamed: 0',
        'D19_LETZTER_KAUF_BRANCHE',
        'EINGEFUEGT_AM',
        # 'OST_WEST_KZ',
        # 'LNR',
        'ALTER_KIND1',
        'ALTER_KIND2',
        'ALTER_KIND3',
        'ALTER_KIND4'
    ], axis=1, inplace=True)


    # convert the the missing values to np.nan
    df_test = convert_toNaN(df_test)
    df_test = impute_simple(df_test, strategy='most_frequent')
    # convert the categorical to numbers
    df_test = encode_labels(df_test, 'CAMEO_DEU_2015')
    df_test = encode_labels(df_test, 'OST_WEST_KZ')

    LNR = df_test['LNR'].astype('int32')

    y_preds = pipeline.predict_proba(df_test.drop('LNR', axis=1))

    predictions = pd.DataFrame(columns=['LNR', 'RESPONSE'])
    predictions['LNR'] = LNR
    predictions['RESPONSE'] = y_preds[:, 1]

    print(predictions.head())
    predictions.to_csv('data/kaggle.csv', index=False)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    # scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # print('Score: ', scores.mean())
    # print(pipeline.get_params())
    #
    # params = {
    #     'over__sampling_strategy': [0.1, 0.15, 0.2],
    #     'under__sampling_strategy': [0.5, 0.6, 0.7],
    #     'cls__n_estimators': [200, 300, 400]
    # }
    #
    # cv = GridSearchCV(pipeline, param_grid=params, scoring='roc_auc', n_jobs=-1)
    # cv.fit(X_train, y_train)
    #
    # print(cv.best_params_)
    # print(cv.best_score_)


if __name__ == '__main__':

    p = train()
    test(p)

    # df_train = pd.read_csv('mailout_train.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'}, sep=',')
    # # df_azdias = pd.read_csv('data/azdias_corrected.csv', dtype={'CAMEO_DEUG_2015': 'str', 'CAMEO_INTL_2015': 'str'},
    # #                         sep=',')
    # pd.set_option('display.max_rows', df_train.shape[1] + 1)
    #
    # # print(sorted(Counter(df_train)))
    # # print(sorted(Counter(df_azdias)))
    # print(df_train.describe(exclude=np.number))
    # df_train.drop([
    #     'Unnamed: 0',
    #     'D19_LETZTER_KAUF_BRANCHE',
    #     'EINGEFUEGT_AM',
    #     # 'OST_WEST_KZ',
    #     'LNR',
    #     'ALTER_KIND1',
    #     'ALTER_KIND2',
    #     'ALTER_KIND3',
    #     'ALTER_KIND4'
    # ], axis=1, inplace=True)
    #
    # print(df_train.columns)
    # # convert the the missing values to np.nan
    # df_train = convert_toNaN(df_train)
    # # impute the missing values
    # # df_train['CAMEO_DEU_2015'] = impute_simple(df_train['CAMEO_DEU_2015'].to_numpy().reshape(-1, 1), columns=['CAMEO_DEU_2015'], strategy='most_frequent')
    # # df_train = encode_labels(df_train, 'CAMEO_DEU_2015')
    # df_train = impute_simple(df_train, strategy='most_frequent')
    # # convert the categorical to numbers
    # df_train = encode_labels(df_train, 'CAMEO_DEU_2015')
    # df_train = encode_labels(df_train, 'OST_WEST_KZ')
    # # df_train = encode_labels(df_train, 'D19_LETZTER_KAUF_BRANCHE')
    #
    # # df_train['EINGEZOGENAM_HH_JAHR'] = 0.01 * df_train['EINGEZOGENAM_HH_JAHR']
    # # df_train['GEBURTSJAHR'] = 0.01 * df_train['GEBURTSJAHR']
    # # df_train['KBA13_ANZAHL_PKW'] = 0.01 * df_train['KBA13_ANZAHL_PKW']
    # # df_train['MIN_GEBAEUDEJAHR'] = 0.01 * df_train['MIN_GEBAEUDEJAHR']
    #
    # # print(Counter(df_train['RESPONSE']))
    #
    # # cols = list(np.setdiff1d(np.array(df_train.columns), np.array(df_azdias.columns)))
    # # cols.remove('RESPONSE')
    # # print(cols)
    # # df_train.drop(cols, axis=1, inplace=True)
    # # df_train = scale(df_train)
    # # print(sorted(Counter(df_train)))
    #
    # X = df_train.drop('RESPONSE', axis=1)
    # # X = scale(X)
    # y = df_train['RESPONSE'].copy().astype('int')
    #
    # print(Counter(y))
    #
    # # X, y = oversample(X, y, strategy=0.2)
    # # X, y = undersample(X, y, strategy=0.4)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #
    # # X_train, y_train = oversample(X_train, y_train, strategy=0.4)
    #
    # print(df_train.shape)
    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)
    #
    # print(Counter(y_train))
    #
    # # df_train = convert_toNaN(df_train)
    # # df_train = impute_simple(df_train, strategy='most_frequent')
    # # print(df_train.isnull().sum())
    # #
    # # print(df_train.shape)
    # # print(df_azdias.shape)
    # # cols = np.setdiff1d(np.array(df_train.columns), np.array(df_azdias.columns))
    # # df_train.drop(cols, axis=1, inplace=True)
    # # print(df_train.shape)
    # #
    # # df_train.fillna(np.nan)
    # # print(df_train.isnull().sum())
    #
    # scaler = StandardScaler()
    # over = RandomOverSampler(sampling_strategy=0.1)
    # under = RandomUnderSampler(sampling_strategy=0.4)
    # # cls = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # cls = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    #
    # steps = [
    #     ('scaler', scaler),
    #     ('over', over),
    #     ('under', under),
    #     ('cls', cls)
    # ]
    #
    # pipeline = Pipeline(steps=steps)
    #
    # pipeline.fit(X_train, y_train)
    #
    # y_preds = pipeline.predict_proba(X_test)
    #
    # score = roc_auc_score(y_test, y_preds[:, 1])
    # print(score)
    #
    # # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    # # scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # # print('Score: ', scores.mean())
    # # print(pipeline.get_params())
    # #
    # # params = {
    # #     'over__sampling_strategy': [0.1, 0.15, 0.2],
    # #     'under__sampling_strategy': [0.5, 0.6, 0.7],
    # #     'cls__n_estimators': [200, 300, 400]
    # # }
    # #
    # # cv = GridSearchCV(pipeline, param_grid=params, scoring='roc_auc', n_jobs=-1)
    # # cv.fit(X_train, y_train)
    # #
    # # print(cv.best_params_)
    # # print(cv.best_score_)

