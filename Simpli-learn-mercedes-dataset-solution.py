# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:41:55 2019

@author: 240022854
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold as vt
from sklearn.preprocessing import LabelEncoder as le
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import xgboost as xgb

data =  pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X = pd.DataFrame(data)
y= X.y.copy()
X.drop(['y'], axis=1, inplace=True)
X
ID = X.ID.copy()
ID

test_data.drop(['ID'],axis=1, inplace=True)

encode_X = X.select_dtypes(include=['object'])
encode_X
other_X=X.copy()
other_X.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1, inplace=True)
other_X
selector = vt()
other_df = selector.fit_transform(other_X)
other_df= pd.DataFrame(other_df)
other_df #removed the variable with variance zeri
ID=ID.to_frame()
type(ID)
y=y.to_frame()
type(y)

encode_test_data = test_data.select_dtypes(include=['object'])
other_test_data=test_data.copy()
other_test_data.drop(['X0','X1','X2','X3','X4','X5','X6','X8'],axis=1, inplace=True)
selector = vt()
other_test_df = selector.fit_transform(other_test_data)
other_test_df= pd.DataFrame(other_test_df)

encode_X = encode_X.apply(le().fit_transform)  #Label encoding
encode_X

encode_test_data = encode_test_data.apply(le().fit_transform)


X_train = pd.concat([encode_X,other_df], axis=1)
X_train.notnull()
X_train.isnull()

X_test = pd.concat([encode_test_data,other_test_df],axis=1)
X_test.notnull()
X_test.isnull()

X_train
#applying PCA on dataset

scaler = MinMaxScaler(feature_range=[0,1])
data_rescaled = scaler.fit_transform(X_train)
pca = PCA().fit(data_rescaled)
pca
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Mercedes')
plt.show()

X_test_reduced = scaler.fit_transform(X_test)
pca1 = PCA().fit(X_test_reduced)

pca = PCA(n_components=100)
X_reduced = pca.fit_transform(data_rescaled)
X_reduced = pd.DataFrame(X_reduced)

X_test_reduced = pca1.fit_transform(X_reduced) 
type(X_test_reduced)


#xgboost
xgb_model = xgb.XGBClassifier().fit(X_reduced,y)

X_test_reduced1 = pd.DataFrame(X_test_reduced)
pred = xgb_model.predict(X_test_reduced1)
  