import glob
import os

from imblearn.metrics import classification_report_imbalanced
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import normalize
from imblearn import pipeline as pl

from classifiers import classifiers
from datasets import datasets
from encoders import encoders
from oversampling import oversampling_methods


def Run():
	for filename in datasets:
		filename = './../input/' + filename + '.csv'
		print(filename)
		df_data = pd.read_csv(filename, header=None)
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
		
		Y = np.array(df_data.iloc[:, -1])
		X = normalize(np.array(df_data.iloc[:, 0:-1]))
		skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
		for train_index, test_index in skf.split(X, Y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			for name, clf in classifiers.items():
				print('Ogirinal')
				print(name)
				y_pred = clf.fit(X_train, y_train).predict(X_test)
				print(
						classification_report_imbalanced(y_test, y_pred,
						                                 digits=4))
				# print('SMOTENC')
				# smotenc = pl.make_pipeline(
				# SMOTENC(categorical_features=nominal), SVC())
				# y_pred = smotenc.fit(X_train, y_train).predict(X_test)
				# print(classification_report_imbalanced(y_test, y_pred,
				# digits=4))
				
				for name_encoder, encoder in encoders.items():
					print(name_encoder)
					for name_oversampling, osm in \
							oversampling_methods.items():
						print(name_oversampling)
						pipe = pl.make_pipeline(encoder, osm, clf)
						# Train the classifier with balancing
						pipe.fit(X_train, y_train)
						# Test the classifier and get the prediction
						y_pred = pipe.predict(X_test)
						# Show the classification report
						print(classification_report_imbalanced(y_test, y_pred,
						                                       digits=4))
