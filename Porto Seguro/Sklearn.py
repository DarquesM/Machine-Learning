# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:12:52 2017

Using SKLEARN

@author: mdarq
"""


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
#%%
data = pd.read_csv("./train.csv")
features = np.array(data.columns)
target = "target"
# Remove first column and target
features = np.delete(features,[0,1])
#%%

class encoding:
    """
    This will apply sklearn.preprocessing.LabelBinarizer and sklearn.preprocessing.StandardScaler
    :example:
    >>> Test
    Blabla
    >>> Other test
    TTT blabla
    
    :param data: A pandas DataFrame

    """

#    def __init__(self, binary=list(), categorical=list(), numeric=list()):
    def __init__(self, binary=list(), categorical=list(), numeric=list()):
        #binary, categorical or numerical features
        self.binary = binary
        self.categorical = categorical
        self.numeric = numeric
    
    def __repr__(self):
        return "<Binarizer and Scaler> object"
    
    def get_cat(self,data):
        colnames = data[features].columns
        for col in colnames:
            c = col.split('_')[-1]
            if c == "cat":
                self.categorical.append(col)                                
            elif c == "bin":
                self.binary.append(col)             
            elif (c != "cat")&(c != "bin"):
                self.numeric.append(col)
        return self.categorical, self.binary, self.numeric
    
    def NA_encode(self,data):
        self.get_cat(data)
        #Implement 'most frequent' strategy
        print("Processing categorical features...")
        for c in self.categorical:
            #Null values are represented by "-1", switching to NaN
            print("Number of missing values for feature: {0}".format(c))
            data.loc[(data[c] == -1, c)] = np.NaN
            na_count = data[c].isnull().sum()
            data[c].fillna(value=data[c].mode()[0], inplace=True)
            print(na_count) #Checking number of null values
        
        #Implement 'median' strategy
        print("Processing numerical features...")
        for c in self.numeric:
            print("Number of missing values for feature: {0}".format(c))
            data.loc[(data[c] == -1, c)] = np.NaN
            na_count = data[c].isnull().sum()
            print(na_count) #Checking number of null values
            data[c].fillna(value=data[c].median(), inplace=True)

    def Binarize_scale(self,data):
        """
        Get column names and determine if numeric/cat/bin
        :param data: A pandas DataFrame
        """
        
        self.get_cat(data)
        print("Enconding null values")
        self.NA_encode(data)
        print("Binarizing...")
#       Let's first create the LabelBinarized features              
        label = LabelBinarizer()
        for c in self.categorical:   
            _ = label.fit_transform(data[c])
            for i in range(np.shape(_)[1]):              
               data[str(c)+"_"+str(i)] = _[:,i]
        print("Scaling...")
#       Scale numeric features
        scaler = StandardScaler()
        for c in self.numeric:
            _ = scaler.fit_transform(np.float64(data[c]).reshape(-1,1))
            
        return None
        
     
b = encoding()
#b.Binarize_scale(data)
b.Binarize_scale(data)
#A = b.Binarize_scale(data)
#print("Data contains {0} categorical, {1} numerical and {2} binary features".format(len(b.categorical), len(b.numeric), len(b.binary)))

#%%
features = np.array(data.columns)
target = "target"
# Remove first column and target
features = np.delete(features,[0,1])

train, test = train_test_split(data)
X_train, y_train, X_test, y_test = train[features], train[target], test[features], test[target]

logit_model = LogisticRegression(n_jobs=1, class_weight={.1:.9})
t_init = time.time()
logit_model.fit(X_train, y_train)
t_final = time.time()
print("Total time: {0}s".format(t_final-t_init))

print(logit_model.score(X_train,y_train))
print(logit_model.score(X_test,y_test))
pred = logit_model.predict(X_test)
#%%
#SUBMISSION
#Data for submission
data_valid = pd.read_csv("test.csv")
c = encoding().Binarize_scale(data_valid)
pred = logit_model.predict(data_valid[features])
#%%
df = pd.DataFrame()
df["id"] = data_valid["id"]
df["target"] = pred

df.to_csv("pred1.csv", index=None, sep=",")
