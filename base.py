# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t3Q7b9Zc3OOovvDYI5PWNdq34iIdjx7U
"""

import category_encoders as ce
import pandas as pd
from sklearn.datasets import fetch_openml
from imblearn import pipeline as pl
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

ds = fetch_openml("Australian")
y = ds.target.astype(int)
X = pd.DataFrame(ds.data, columns=ds.feature_names)

nominal = [0,3,4,5,7,8,10,11]
nominal_names = X.columns[nominal]

encoders = {'BackwardDifference': ce.BackwardDifferenceEncoder(cols=nominal_names),
            'BaseNEncoder': ce.BaseNEncoder(cols=nominal_names),
            'BinaryEncoder': ce.BinaryEncoder(cols=nominal_names),
            'CatBoostEncoder': ce.CatBoostEncoder(cols=nominal_names),
            'CountEncoder': ce.CountEncoder(cols=nominal_names),
            'GLMMEncoder': ce.GLMMEncoder(cols=nominal_names),
            'HashingEncoder': ce.HashingEncoder(cols=nominal_names),
            'HelmertEncoder': ce.HelmertEncoder(cols=nominal_names),
            'JamesSteinEncoder': ce.JamesSteinEncoder(cols=nominal_names),
            'LeaveOneOutEncoder': ce.LeaveOneOutEncoder(cols=nominal_names),
            'MEstimateEncoder': ce.MEstimateEncoder(cols=nominal_names),
            'OneHotEncoder': ce.OneHotEncoder(cols=nominal_names),
            'OrdinalEncoder' : ce.OrdinalEncoder(cols=nominal_names),
            'SumEncoder': ce.SumEncoder(cols=nominal_names),
            'PolynomialEncoder': ce.PolynomialEncoder(cols=nominal_names),
            'TargetEncoder': ce.TargetEncoder(cols=nominal_names),
            'WOEEncoder': ce.WOEEncoder(cols=nominal_names)}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print('Ogirinal')

y_pred = SVC().fit(X_train,y_train).predict(X_test)
print(classification_report_imbalanced(y_test, y_pred,digits=4))

print('SMOTENC')


smotenc = pl.make_pipeline(SMOTENC(categorical_features = nominal ),SVC())

y_pred = smotenc.fit(X_train,y_train).predict(X_test)
print(classification_report_imbalanced(y_test, y_pred,digits=4))


for name, encoder in encoders.items():
  print(name)
  pipe = pl.make_pipeline(encoder,SMOTE(),SVC())
  
  # Train the classifier with balancing
  pipe.fit(X_train, y_train)

  # Test the classifier and get the prediction
  y_pred = pipe.predict(X_test)

  # Show the classification report
  print(classification_report_imbalanced(y_test, y_pred,digits=4))