# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:57:53 2017

@author: Michael
"""

import mlbox as mlb
import numpy as np
import pandas as pd
import time
#%%
data_valid = pd.read_csv("./2-Prepared Data/data_validation_prep.csv")
answer = pd.DataFrame()
answer["Id"] = (data_valid.Id).astype(int)
#%%
# reading and cleaning the train and test files
#Unprocessed data
df=mlb.preprocessing.Reader(sep=",").train_test_split(["./1-Original Data/train.csv","./1-Original Data/test.csv"],'SalePrice')
#Manually pre processed Data
#df=mlb.preprocessing.Reader(sep=",").train_test_split(["./2-Prepared Data/data_prep.csv","./2-Prepared Data/data_validation_prep.csv"],'SalePrice')

target_name = "SalePrice"
#%%
df=mlb.preprocessing.Drift_thresholder().fit_transform(df)

#%%
#Récupérer les données transformées pour appliquer mes propres modèles
#df['train'].to_csv("data_train.csv", index=False)
dataTrain=df["train"]
dataTrain["SalePrice"] = df["target"]
dataTest = df["test"]
dataTest["Id"]=(data_valid.Id).astype(int)
dataTrain.to_csv("data_train.csv", index=False)
dataTest.to_csv("data_test.csv",index=False)
#%%
""" 
**********************
****** STACKING ******
**********************
"""
params = { 'stck__base_estimators' : [Regressor(strategy = "RandomForest"), Regressor(strategy = "XGBoost")],
   
                                      "est__strategy" : "Linear" }
'''
params = { 'stck1__base_estimators' : [Regressor(strategy = "RandomForest"), Regressor(strategy = "XGBoost")],
                                      'stck2__base_estimators' : [Regressor(strategy = "ExtraTrees"),
                                                                Regressor(strategy = "LightGBM")], "est__strategy" : "Linear" }
'''

#%%
""" 
*********************
****** XGBOOST ******
*********************
"""
# removing the drift variables
# fs = feature selection
# ce = categorical encoder

t_init = time.time()

# setting the hyperparameter space
# Equivalent to a pipeline
#ce_strategy : "None" pour remplacer les missing values
space_xgb={
'ne__numerical_strategy'    :{"search":"choice",
                              "space":[0,'mean','median','most_frequent']},
'ne__categorical_strategy'  :{"search":"choice",
                              "space":[np.NaN,"None"]},
'ce__strategy'              :{"search":"choice",
                              "space":['label_encoding','entity_embedding','dummification']},
'fs__strategy'              :{"search":"choice",
                              "space":['l1','variance','rf_feature_importance']},
'fs__threshold'             :{"search":"uniform",
                              "space":[0.01,0.6]},
'est__strategy'             :{"search":"choice",
                              "space":["XGBoost"]},
'est__max_depth'            :{"search":"choice",
                              "space":[3,4,5,6,7]},
'est__learning_rate'        :{"search":"uniform",
                              "space":[0.01,0.1]},
'est__subsample'            :{"search":"uniform",
                              "space":[0.4,0.9]},
'est__reg_alpha'            :{"search":"uniform",
                              "space":[0,10]},
'est__reg_lambda'           :{"search":"uniform",
                              "space":[0,10]},
'est__n_estimators'         :{"search":"choice",
                              "space":[1000,1250,1500]}
}

# calculating the best hyper-parameter
best_xgb=mlb.optimisation.Optimiser(scoring="r2",n_folds=5).optimise(space_xgb,df,120)
#DeprecationWarning: Scoring method mean_squared_error was renamed to neg_mean_squared_error in version 0.18 and will be removed in 0.20.
#Available scorings for regression : “mean_absolute_error”, “mean_squared_error”, “median_absolute_error”, “r2”
t_final=time.time()
print("Total execution time : {}s".format(t_final-t_init))
#2600 secondes avec 100 iterations

# predicting on the test dataset for XGBoost
mlb.prediction.Predictor().fit_predict(best_xgb,df)

data = pd.read_csv("./save/SalePrice_predictions.csv")
answer["SalePrice_XGB"] = data["SalePrice_predicted"]

#%%
""" 
******************
****** LGBM ******
******************
"""
# removing the drift variables
# fs = feature selection
# ce = categorical encoder

t_init = time.time()
# setting the hyperparameter space
# Equivalent to a pipeline
#ce_strategy : "None" pour remplacer les missing values
space_lgbm={
'ne__numerical_strategy'    :{"search":"choice" ,
                                      "space"     :[0,'mean','median','most_frequent']},
'ne__categorical_strategy'  :{"search":"choice" ,
                                      "space"     :[np.NaN,"None"]},
'ce__strategy'              :{"search":"choice" ,
                                      "space"     :['label_encoding','entity_embedding','dummification']},
'fs__strategy'              :{"search":"choice" ,
                                      "space"     :['l1','variance','rf_feature_importance']},
'fs__threshold'             :{"search":"uniform",
                                      "space"     :[0.01,0.6]},
'est__strategy'             :{"search":"choice" ,
                                      "space"     :["LightGBM"]},
'est__reg_alpha'            :{"search":"uniform",
                                      "space"     :[0,4]},
'est__reg_lambda'           :{"search":"uniform",
                                      "space"     :[0,4]},
'est__subsample'            :{"search":"uniform",
                                      "space"     :[0.7,0.95]},
'est__learning_rate'        :{"search":"uniform",
                                      "space"     :[0.01,0.5]}
}

# calculating the best hyper-parameter
best_lgbm=mlb.optimisation.Optimiser(scoring="neg_mean_squared_error",n_folds=5).optimise(space_lgbm,df,40)
#DeprecationWarning: Scoring method mean_squared_error was renamed to neg_mean_squared_error in version 0.18 and will be removed in 0.20.
#Available scorings for regression : “mean_absolute_error”, “mean_squared_error”, “median_absolute_error”, “r2”

mlb.prediction.Predictor().fit_predict(best_lgbm,df)

t_final=time.time()
print("Total execution time : {}s".format(t_final-t_init))

#reg_alpha ? N'est pas dans la documentation de LightGBM
#Dans la documentation : class mlbox.preprocessing.Drift_thresholder(threshold=0.8, inplace=False, verbose=True, to_path='save') mais on a threshold : float (between 0.5 and 1.), defaut = 0.9
#Attention aux espaces dans "est__XXXX" (ne pas écrire 'est__subsample ') par exp
#n_estimators : si on y touche, impossible de fit le pipeline

#Read prediction file
data = pd.read_csv("./save/SalePrice_predictions.csv")
#Insert it into dataframe
answer["SalePrice_LGBM"] = data["SalePrice_predicted"]
#%%
#Averaging answers 
answer["Ensemble"] = (answer["SalePrice_RF"]+answer["SalePrice_XGB"])/2

#%%


final_answer = pd.DataFrame()
final_answer["Id"] = (data_valid.Id).astype(int)
final_answer["SalePrice"] = answer["SalePrice_XGB"]
final_answer.to_csv("answer-SalePrice_XGB.csv",index=False)




#%%

"""
Default parameters : 0.26013
AVidhya : 0.2151
XGBoost : 0.14818
XGBoost : 0.13146 (rank 818)
XGBoost + RandomForest = 0.13276
RandomForest = 0.15196
Bagging : 0.15099
XGBoost+Bagging+RandomForest : 0.13581
XGBoost new : 0.12867 (rank 750)
LGBM : 0.26004 (default parameters)
XGBoost : 0.12672 (optimized parameters and 100 iterations)
XGBoost : 0.12827 (optimized parameters and 100 iterations, + manual preprocessing remove : ["GarageYrBlt", "MoSold","MasVnrArea", "GarageCars"], create AgeTRavaux)
XGBoost : 0.12852 (optimized parameters and 100 iterations, + manual preprocessing remove :["MoSold","GarageCars", "YearRemodAdd"], create AgeTRavaux)
"""

#%% 
""" *** RANDOM FOREST *** """
#df=mlb.preprocessing.Drift_thresholder().fit_transform(df)

space_RF={'ne__numerical_strategy':{"search":"choice","space":[0,'mean','median']},
'ne__categorical_strategy':{"search":"choice","space":[np.NaN,"None"]},
'ce__strategy':{"search":"choice","space":['label_encoding','entity_embedding','dummification']},
'fs__strategy':{"search":"choice","space":['l1','variance','rf_feature_importance']},
'fs__threshold':{"search":"uniform","space":[0.01, 0.7]},
'est__strategy': {"search":"choice", "space":["RandomForest"]},
'est__max_depth':{"search":"choice","space":[2,3,5,7,10]},
'est__min_samples_split':{"search":"uniform","space":[2,10]},
'est__n_estimators':{"search":"uniform","space":[10,200]}}
best_RF=mlb.optimisation.Optimiser(scoring="neg_mean_squared_error",n_folds=5).optimise(space_RF,df,40)
mlb.prediction.Predictor().fit_predict(best_RF,df)
#%%
data = pd.read_csv("./save/SalePrice_predictions.csv")
answer["SalePrice_RF"] = data["SalePrice_predicted"]
