# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:21:31 2017

@author: mdarq
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, StandardScaler

class encoder:
    """
    Applies sklearn.preprocessing.LabelBinarizer and 
    sklearn.preprocessing.StandardScaler
    """

    def __init__(self):
        #binary, categorical or numerical features
        self.binary = list()
        self.categorical = list()
        self.numeric = list()
        self.feat_remove = list()
        self.binarizer = False

    def __repr__(self):
        return "<Binarizer and Scaler> object"
    
    def get_params(self):
        
        """ Returns parameters 
        
        Parameters
        ----------
        None
        
        Return
        ------
        Dictionnary : Keywords
        """
        
        return {"Binarizer": self.binarizer}

    def auto_cat(self,df):        
        ''' 
        Reads DataFrame to automatically determine if a feature is binary, 
        categorical, numerical or other.
        Not taking into account texte, dates, other...
        
        Parameters
        ----------
        df : Pandas DataFrame
             DataFrame whose columns are to be categorized
             
        Returns
        ------
        None
        
        '''
        colnames = df.columns
        for col in colnames:
            if (col=='target') | (col=='id'):
                pass
            # If only 2 different values then binary 
            elif len(pd.unique(df[col]))==2:
                self.binary.append(col)
                print("{0}\t-\tBinary".format(str(col)))           
            # If integer then categorical
            elif np.dtype(df[col])==np.int64:
                ''' if integer and shorter than total length then categorical
                 could be index 
                '''
                self.categorical.append(col)
                print("{0}\t-\tCategorical".format(str(col)))
            # Everything else is considered as a numeric feature
            else:
                print("{0}\t-\tNumeric".format(str(col)))
                self.numeric.append(col)
    
    def na_encode(self,df, NA=np.NaN):
        ''' 
        
        Encodes null values using different strategies depending on the 
        category of the columns
        
        Parameters
        ----------
        df : Pandas DataFrame
            DataFrame containing the data
        NA : The representation of missing values in the DataFrame
        
        Returns
        -------
        df : Pandas DataFrame
            The DataFrame after its missing values were replaced
        '''
        
        #Implement 'most frequent' strategy
        print("Processing categorical features...")
#        _df = df.copy()
#        self.auto_cat(df)
        for c in self.categorical:
            #Null values are represented by "-1", switching to NaN
            print("Number of missing values for feature: {0}".format(c))
            df.loc[(df[c] == NA, c)] = np.NaN
            na_count = df[c].isnull().sum()
            df[c].fillna(value=df[c].mode()[0], inplace=True)
            print(na_count) #Checking number of null values            
        #Implement 'mean' or 'median" strategy
        print("Processing numerical features...")
        for c in self.numeric:
            #NaN are represented as "-1" in the original data
            df.loc[(df[c] == NA, c)] = np.NaN
            na_count = df[c].isnull().sum()
            print("Number of missing values for feature: {0}\n{1}".format(c,
                  na_count))
            #Replace NaN values
            df[c].fillna(value=df[c].mean(), inplace=True)
        for c in self.binary:
            print("Number of missing values for feature: {0}".format(c))
            df.loc[(df[c] == NA, c)] = np.NaN
            na_count = df[c].isnull().sum()
            df[c].fillna(value=df[c].mode()[0], inplace=True)
            print(na_count) #Checking number of null values            
        return df

    def binarize_scale(self, df, NA=np.NaN, binarizer=False):
        
        """
        Get column names and determine if numeric/cat/bin
        
        parameters
        ----------
        
        df: A pandas DataFrame
            DataFrame containing the data that will be processed
        NA : The representation of missing values in the DataFrame
        binarizer : Boolean
            Select if the categorical features will be binarized or not
        
        Returns
        -------
        df : Pandas DataFrame
            The DataFrame after processing (binarized and scaled)
        """
        
        self.auto_cat(df)
        print("Encoding null values")
        df = self.na_encode(df,NA)
        self.binarizer = binarizer
        if self.binarizer == True:
            print("Binarizing...")
            #       Let's first create the LabelBinarized features                      
            label = LabelBinarizer()
            for c in self.categorical:   
                print(c)
                _ = label.fit_transform(df[c])
                for i in range(np.shape(_)[1]):              
                    df[str(c)+"_"+str(i)] = _[:,i]
        print("Scaling...")
#       Scale numeric features
        scaler = StandardScaler()
        for c in self.numeric:
            _ = scaler.fit_transform(np.float64(df[c]).reshape(-1,1))
        return df