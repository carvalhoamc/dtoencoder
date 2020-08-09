import glob
import os

from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import arff

from datasets import datasets


def createValidationData(folder):
        """
        Create sub datasets for cross validation purpose
        :param datasets: List of datasets
        :param folder: Where datasets was stored
        :return:
        """
        for filename in datasets:
            filename = './../input/'+filename+'.csv'
            print(filename)
            df_data = pd.read_csv(filename,header=None)
            drop_na_col = True,  ## auto drop columns with nan's (bool)
            drop_na_row = True,  ## auto drop rows with nan's (bool)
             ## pre-process missing values
            if bool(drop_na_col) == True:
                df_data = df_data.dropna(axis=1)  ## drop columns with nan's

            if bool(drop_na_row) == True:
                df_data = df_data.dropna(axis=0)  ## drop rows with nan's

            ## quality check for missing values in dataframe
            if df_data.isnull().values.any():
                raise ValueError("cannot proceed: data cannot contain NaN values")
            
            print(df_data.shape)

            Y = np.array(df_data.iloc[:,-1])
            X = normalize(np.array(df_data.iloc[:,0:-1]))
            skf = StratifiedShuffleSplit(n_splits=5,)
            dataset = filename.replace(folder, "")
            dataset = dataset.replace('.arff','')
            for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                y_train = y_train.reshape(len(y_train), 1)
                y_test = y_test.reshape(len(y_test), 1)
                train = pd.DataFrame(np.hstack((X_train, y_train)))
                test = pd.DataFrame(np.hstack((X_test, y_test)))
                os.makedirs(os.path.join(folder, dataset, str(fold)))
                train.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_train.csv"])), header=False,
                             index=False)
                test.to_csv(os.path.join(folder, dataset, str(fold), ''.join([dataset, "_test.csv"])), header=False,
                            index=False)