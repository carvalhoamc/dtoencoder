import glob
import os

from category_encoders import OneHotEncoder, TargetEncoder, OrdinalEncoder
from category_encoders.utils import is_category
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTENC
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import normalize, LabelEncoder
from imblearn import pipeline as pl

from classifiers import classifiers
from datasets import datasets
from encoders import encoders
from oversampling import oversampling_methods


def get_obj_cols(df):
	""" Returns names of 'object' columns in the DataFrame. """
	obj_cols = []
	for idx, dt in enumerate(df.dtypes):
		if dt == 'object' or is_category(dt):
			obj_cols.append(df.columns.values[idx])
	return obj_cols


def run():
	dfcol = ['ID', 'DATASET', 'FOLD', 'PREPROC', 'ALGORITHM', 'ORDER', 'ALPHA',
	         'ENCODER', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA']
	df = pd.DataFrame(columns=dfcol)
	i = 0
	
	for filename in datasets:
		fname = './../input/' + filename + '.csv'
		print(fname)
		df_data = pd.read_csv(fname, header=None)
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
		
		categorical_cols = get_obj_cols(df_data.iloc[:, 0:-1])
		Y = np.array(df_data.iloc[:, -1])
		#X = normalize(np.array(df_data.iloc[:, 0:-1]))
		X = np.array(df_data.iloc[:, 0:-1])
		le = LabelEncoder()
		Y = le.fit_transform(Y)
		
		
		skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
		fold = 0
		for train_index, test_index in skf.split(X, Y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			y_train = y_train.reshape(len(y_train), 1)
			y_test = y_test.reshape(len(y_test), 1)
			print('Folder = ', fold)
			
			for name, clf in classifiers.items():
				print('Ogirinal')
				print(name)
				target = OneHotEncoder(cols=categorical_cols)
				encoded_train = target.fit_transform(X_train, y_train)
				encoded_test = target.transform(X_test, y_test)
				clf.fit(encoded_train, y_train)
				y_pred = clf.predict(encoded_test)
				res = classification_report_imbalanced(y_test, y_pred,
				                                       digits=4)
				identificador = filename + '_' + 'original' + '_' + name
				aux = res.split()
				score = aux[-7:-1]
				df.at[i, 'ID'] = identificador
				df.at[i, 'DATASET'] = filename
				df.at[i, 'FOLD'] = fold
				df.at[i, 'PREPROC'] = 'original'
				df.at[i, 'ALGORITHM'] = name
				df.at[i, 'ORDER'] = 'NONE'
				df.at[i, 'ALPHA'] = 'NONE'
				df.at[i, 'ENCODER'] = 'NONE'
				df.at[i, 'PRE'] = score[0]
				df.at[i, 'REC'] = score[1]
				df.at[i, 'SPE'] = score[2]
				df.at[i, 'F1'] = score[3]
				df.at[i, 'GEO'] = score[4]
				df.at[i, 'IBA'] = score[5]
				i = i + 1
				
				print('SMOTENC')
				
				smotenc = pl.make_pipeline(
					SMOTENC(categorical_features=categorical_cols),clf)
				y_pred = smotenc.fit(encoded_train, y_train).predict(encoded_test)
				res = classification_report_imbalanced(y_test, y_pred,
				                                       digits=4)
				
				identificador = filename + '_' + 'smotenc' + '_' + name
				aux = res.split()
				score = aux[-7:-1]
				df.at[i, 'ID'] = identificador
				df.at[i, 'DATASET'] = filename
				df.at[i, 'FOLD'] = fold
				df.at[i, 'PREPROC'] = 'smotenc'
				df.at[i, 'ALGORITHM'] = name
				df.at[i, 'ORDER'] = 'NONE'
				df.at[i, 'ALPHA'] = 'NONE'
				df.at[i, 'ENCODER'] = 'NONE'
				df.at[i, 'PRE'] = score[0]
				df.at[i, 'REC'] = score[1]
				df.at[i, 'SPE'] = score[2]
				df.at[i, 'F1'] = score[3]
				df.at[i, 'GEO'] = score[4]
				df.at[i, 'IBA'] = score[5]
				i = i + 1
				
				
			for name, clf in classifiers.items():
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
						res = classification_report_imbalanced(y_test, y_pred,
						                                       digits=4)
						identificador = filename + '_' + name_encoder + '_' + name_oversampling + '_' + name
						aux = res.split()
						score = aux[-7:-1]
						df.at[i, 'ID'] = identificador
						df.at[i, 'DATASET'] = filename
						df.at[i, 'FOLD'] = fold
						df.at[i, 'PREPROC'] = name_oversampling
						df.at[i, 'ALGORITHM'] = name
						if name_oversampling == 'dtosmote':
							df.at[i, 'ORDER'] = 'solid_angle'
							df.at[i, 'ALPHA'] = 7.0
						else:
							df.at[i, 'ORDER'] = 'NONE'
							df.at[i, 'ALPHA'] = 'NONE'
						df.at[i, 'ENCODER'] = name_encoder
						df.at[i, 'PRE'] = score[0]
						df.at[i, 'REC'] = score[1]
						df.at[i, 'SPE'] = score[2]
						df.at[i, 'F1'] = score[3]
						df.at[i, 'GEO'] = score[4]
						df.at[i, 'IBA'] = score[5]
						i = i + 1
					df.to_csv('./../output/encoder_results.csv', index=False)
			fold = fold + 1
