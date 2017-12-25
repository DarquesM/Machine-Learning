# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:29:54 2017

@author: mdarq

Features selection with ExtraTressClassifier
PCA
Logistic Regression

"""

import time

from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,\
                                    RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score,\
                                matthews_corrcoef

from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import encoder
import tensorflow as tf

import xgboost as xgb

#import seaborn as sns

def removefeatures(df, df2):
    """
    Will delete columns containing useless features (see discussions on kaggle)
    """
    for d in [df,df2]:
        for c in d.columns:
            a = c.split('_')
            if 'calc' in a:
                d.drop(c, inplace=True, axis=1)
    return df,df2

def RFmodel(n_estimators,min_samples_leaf, class_weight):
    '''
    Defining a random forest classifier, Fitting on training set, Prediction
    on test set
    
    Parameters
    ----------
    n_estimators, min_samples_leaf : int
        Hyperparameters for the random forest classifier, see scikit-learn 
        documentation
    
    Returns  
    -------
    train_score : the accuracy score calculated on the training set
    test_score : the accuracy score calculated on the test set
    auc_roc : auc_roc estimated on test set
    RF_model : the Random Forest model for further prediction
    '''
    RF_model = RandomForestClassifier(n_jobs=-1,
                                      n_estimators=n_estimators,
                                      min_samples_leaf=min_samples_leaf,
                                      class_weight=class_weight)
    t_init = time.time()
    print("Fitting train set with\n n_estimators = {0}\n min_samples_leaf ={1}"
          .format(n_estimators, min_samples_leaf))
    RF_model.fit(X_train,y_train)
    print("Total time: {0}s".format(time.time()-t_init))
    train_score = RF_model.score(X_train,y_train)
    test_score = RF_model.score(X_test,y_test)
    print("Train score: {0}".format(train_score))
    print("Test score: {0}".format(test_score))
    pred_RF_test = RF_model.predict(X_test)
    print("Number of positive preditcions on test set: ",pred_RF_test.sum())
    print("There are {:.2f}% of positive predictions on the test test"
      .format(100*pred_RF_test.sum()/len(X_test)))
    auc_roc = roc_auc_score(y_test,pred_RF_test)
    print("AUC-ROC: {0}\n".format(auc_roc))    
    print("MCC score: {0}".format(matthews_corrcoef(y_test,pred_RF_test)))
    return RF_model

def logitmodel(data, C=1, class_weight='balanced', solver='saga', penalty='l1'
               ,max_iter=100, n_jobs=-1, scoring='roc_auc'):
    """
    Trains and fits a logistic regression model. Works when data is simply
    split into 1 train & 1 test set
    
    """
    logit_model = LogisticRegression(n_jobs=n_jobs,
                                 max_iter=max_iter,
                                 class_weight=class_weight,
                                 C=C,
                                 solver=solver,
                                 penalty=penalty,
                                 scoring='roc_auc')
    print("Fitting with parameters:\n {0}".format(logit_model.get_params))
    t_init = time.time()
    logit_model.fit(X_train,y_train)
    t_final = time.time()
    print("Total time: {0}s".format(t_final-t_init))
 
#    Predictions
    pred_logit_test = logit_model.predict(X_test)
    pred_logit_test_proba = logit_model.predict_proba(X_test)[:,1]

#    Scoring
    fpr, tpr, _ = roc_curve(y_test, pred_logit_test, pos_label=1)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    score = logit_model.score(pred_logit_test,)
    print(score)
    print("roc_auc score: {0}"
          .format(logit_model.score(X_test,y_test)))
    print("Somme de positifs sur test: {0}".format(data.sum()))
    print("Somme de positifs sur la prédiction via logistic regression sur\
          test set: {0}".format(pred_logit_test.sum()))
    print("There are {:.2f}% of positive predictions on the test test"
      .format(100*pred_logit_test.sum()/len(data)))
    return logit_model

def pipePCALogitGridSearch(logistic):
    """
    Creates a pipeline that will chain a PCA and a Logit with GridSearchCV
    """
    print("#################\nPCA decomposition\n#################")
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    pca.fit(X_train)
    print("Fitting pipeline")
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components = [150, 200, 250, 300],
                                  logistic__class_weight=[{0:0.1,1:1.2},
                                                          {0:0.05,1:1.2},
                                                          {0:0.15,1:1.2},
                                                          {0:0.2,1:1.2},
                                                          {0:0.25,1:1.2},
                                                          {0:0.3,1:1.2}]),
                             scoring='roc_auc')    
    estimator.fit(X_train,y_train)
    estimator_pred = estimator.predict(X_test)
    print(estimator_pred.sum())
    
#    pred_pipe_PCA_logit = estimator.predict(data_valid[features])
#    pred_pipe_PCA_logit_proba = estimator.predict_proba(data_valid[features])[:,1]
    nb_positive_preds = estimator_pred.sum()
    print("There are {:.2f}% of positive predictions on the test test"
          .format(100*nb_positive_preds/len(test)))
    print("roc_auc score: {0}".format(roc_auc_score(y_test,estimator_pred)))
    if nb_positive_preds !=0:
        print("f1 score: {0}".format(f1_score(y_test,estimator_pred)))
    return pipe

def pipePCALogit(logistic, pca_n_components):
    """
    Creates a pipeline that will chain a PCA and a logit without GridSearch
    Should be used when best hyperparameters were found
    """
    print("#################\nPCA decomposition\n#################")
    pca = decomposition.PCA(n_components=pca_n_components)    
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)],
                           memory='./test')
    print("Fitting PCA with {0} components".format(pca_n_components))
    pca.fit(X_train[features])      
    print("Fitting pipeline")
    pipe.fit(X_train[features],y_train)
    print("Predicting on test set")
    pred_pipe_train = pipe.predict(X_train[features])
    pred_pipe_train_proba = pipe.predict_proba(X_train[features])[:,1]
    pred_pipe_test = pipe.predict(X_test[features])
    pred_pipe_test_proba = pipe.predict_proba(X_test[features])[:,1]
    n_positives_test = pred_pipe_test.sum()
    n_positives_train = pred_pipe_train.sum()
    print("There are {0} positive predictions on the test set"\
          .format(n_positives_test))
    print("There are {0} positive predictions on the train set"\
          .format(n_positives_train))
    print("Predicting on the validation set")
    print("There are {:.2f}% of positive predictions on the train test"
          .format(100*pred_pipe_train.sum()/len(train)))    
    print("There are {:.2f}% of positive predictions on the test test"
          .format(100*pred_pipe_test.sum()/len(test)))
    print("Train set scores:\n",\
          evalscores(y_train,pred_pipe_train,pred_pipe_train_proba))
    print("Test set scores:\n",\
          evalscores(y_test,pred_pipe_test,pred_pipe_test_proba))
    
    print("Predicting on validation data ...")
    pred_pipe_valid = pipe.predict(data_valid_new[features])
    print("This model predicts {0:.2f}% of positive results on the validation data"
              .format(100*pred_pipe_valid.sum()/len(data_valid_new)))
    pred_pipe_valid_proba = pipe.predict_proba(data_valid_new[features])[:,1]
    
    return pipe, pred_pipe_test, pred_pipe_test_proba, pred_pipe_valid_proba

def binarize_encode(dat, dat_valid):
    binarizer = True
    b = encoder.encoder()
    dat = b.binarize_scale(df=dat, NA=-1, binarizer=binarizer)
    c = encoder.encoder()
    dat_valid = c.binarize_scale(df=dat_valid, NA=-1, binarizer=binarizer)
    print("Data contains {0} categorical, {1} numerical and {2} binary features"
          .format(len(b.categorical), len(b.numeric), len(b.binary)))
    print("Validation data contains {0} categorical, {1} numerical and {2} binary\
          features".format(len(c.categorical), len(c.numeric), len(c.binary)))

def submit(prediction,dat_valid,filename):
    df = pd.DataFrame()
    df["id"] = np.int32(data_valid["id"])
    df["target"] = np.float64(prediction).round(decimals=4)
    df.to_csv(filename, index=None, sep=",")

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def evalscores(y_true,y_pred,y_pred_proba):
    '''
    Calculate and display different scores for training, test sets
    '''
    
    rocaucscore = roc_auc_score(y_true,y_pred_proba)
    f1score = f1_score(y_true,y_pred)
    MCCscore = matthews_corrcoef(y_true,y_pred)
#    gini_predictions = gini(y_test, y_pred_proba)
#    gini_max = gini(y_test, y_test)
    ngini= gini_normalized(y_true, y_pred_proba)
    scores={"ROC_AUC":rocaucscore, "f1 score":f1score, "MCC":MCCscore,\
            "Normalized Gini":ngini}
    return scores
    
#%% OPEN FILES
data = pd.read_csv("./train.csv")
data_valid = pd.read_csv("test.csv")

# Drop features with too many missing values
data.drop(["ps_car_03_cat","ps_car_05_cat"], inplace=True, axis=1)
data_valid.drop(["ps_car_03_cat","ps_car_05_cat"], inplace=True, axis=1)
data, data_valid = removefeatures(data, data_valid)

#%% SELECT PROCESS
extratrees = True
binarizer= True
#Choose model from "Logistic", "XGBoost"
selected_model = "Logistic"

#%% FEATURES SELECTION WITH EXTRA TREES
if extratrees==True:
    features_init = data.columns
    features_init = features_init.drop(['id', 'target'])
    target = "target"
    
    clf = ExtraTreesClassifier(n_jobs=-1)
    clf = clf.fit(data[features_init],data[target])
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    #Use threshold to tune the number of selected features
    model = SelectFromModel(clf,threshold=0.003)
    model.fit(data[features_init], data[target])
    #Features selection X_train -> X_train_new
    data_new = model.transform(data[features_init])
    n_kept_features = data_new.shape[1]
    print("The model has kept {0} features".format(n_kept_features))
    
    # Plot the feature importances of the forest
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(data_new.shape[1]), importances[indices][:n_kept_features],
           color="r", yerr=std[indices][:n_kept_features], align="center")
    plt.xticks(range(data_new.shape[1]), indices)
    plt.xlim([-1, data_new.shape[1]])
    plt.show()
    print("LIST OF FEATURES BY IMPORTANCE")
    for i in range(len(features_init)):
        print("feature {0}--\t{1}\t{2}".format(i+1,features_init[indices[i]],
               importances[indices[i]])) 
    
    kept_features = features_init[indices][:n_kept_features]
    
    data_new = pd.DataFrame(data[kept_features].copy())
    #Do not forget to keep the target column
    data_new['target'] = data['target']
    data_valid_new = pd.DataFrame(data_valid[kept_features].copy())
    
    #Binarize
    if binarizer==True:
        binarize_encode(data_new,data_valid_new)
    features = list(data_new.columns)
    features.remove('target')

# Run this only if no preselection with extratrees
if extratrees==False:
    data_new = pd.DataFrame(data).copy()
    data_valid_new=pd.DataFrame(data_valid).copy()
    binarize_encode(data_new,data_valid_new)
    features = list(data_new.columns)
    features.remove('target')
    
#Remove single cat features if they were binarized
if binarizer == True:
    for f in features:
        if (f.split('_')[-1]) == 'cat':
            features.remove(str(f))


#%% Features and train/test split
            
features_data_new = set(data_new.columns)
features_data_valid_new = set(data_valid_new.columns)

#Features only in temp
temp = features_data_new.difference(features_data_valid_new)
for t in temp:
    data_valid_new[t] = np.zeros(len(data_valid_new))
    if t == 'target':
        data_valid_new.drop('target', inplace=True, axis=1)
    
temp = features_data_valid_new.difference(features_data_new)    
for t in temp:
    data_new[t] = np.zeros(len(data_new))

features = list(data_new.columns)
features.remove('target')
if 'id' in features:
    features.remove('id')
target = "target"

train, test = train_test_split(data_new, train_size=0.80, test_size=0.20)

X_train, y_train, X_test, y_test = \
    train[features], train[target], test[features], test[target]
print("There are {:.2f}% of positive predictions in the train set"
      .format(100*y_train.sum()/len(y_train)))
print("There are {:.2f}% of positive predictions in the test set"
      .format(100*y_test.sum()/len(y_test)))

#%%
# =============================================================================
# From here, run only the cells corresponding to the desired model
# =============================================================================


if selected_model == "Logistic":
    logistic = LogisticRegression(n_jobs=-1,
                                     max_iter=1000,
                                     C=1,
                                     class_weight={0:0.1,1:1.2},
    #                                 class_weight='balanced',
                                     solver='saga')
    scores_train=[]
    scores_test=[]
    times=[]
    components = [60, 80,100,120,140,160,180,200,220]
    for c in components:
        t_init=time.time()
        log_model, pred_logit_test, pred_logit_test_proba, pred_logit_valid = \
            pipePCALogit(logistic, c)
        t_final=time.time()
        scores_train.append(gini_normalized(y_train, log_model.predict_proba(X_train)[:,1]))
        scores_test.append(gini_normalized(y_test, log_model.predict_proba(X_test)[:,1]))
        times.append(t_final-t_init)
        print("Temps: {0}\n".format(t_final-t_init))

if selected_model == "XGBoost":
    t_init = time.time()
    xgb_model = xgb.XGBClassifier(n_estimators=500, 
                                  max_depth=3, 
                                  min_child_weight=2,
                                  learning_rate=0.001,
                                  nthread=16,       
                                  subsample=0.9,
                                  reg_alpha=.006,
                                  reg_lambda=.005,
                                  gamma=5,
                                  scale_pos_weight=15
                                  )
    print("Fitting on train set with XGBoost")
    xgb_model.fit(X_train,y_train)
    print("Total fitting time: {}".format(time.time()-t_init))
    print("Predicting on test data")
    xgb_model.predict(X_test)
    t_final = time.time()
    pred_xgb_train = xgb_model.predict(X_train)
    pred_xgb_train_proba = xgb_model.predict_proba(X_train)[:,1]
    pred_xgb_test = xgb_model.predict(X_test)
    pred_xgb_test_proba = xgb_model.predict_proba(X_test)[:,1]
    #Plot ROC AUC curve
    fpr, tpr, _ = roc_curve(y_test, pred_xgb_test_proba, pos_label=1)
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    print("There are {:.2f}% of positive predictions on the test set"
          .format(100*pred_xgb_test.sum()/len(test)))
    print("There are {:.2f}% of positive predictions on the train set"
          .format(100*pred_xgb_train.sum()/len(train)))
    print("Predicting on validation data ...")
    if binarizer == True:
        pred_xgb_valid = xgb_model.predict(data_valid_new[features])
        print("This model predicts {0:.2f}% of positive results on the validation data"
              .format(100*pred_xgb_valid.sum()/len(data_valid_new)))
        pred_xgb_proba_valid = xgb_model.predict_proba(data_valid_new[features])[:,1]
    elif binarizer == False:
        pred_xgb_valid = xgb_model.predict(data_valid[features])
        print("This model predicts {0:.2f}% of positive results on the validation data"
              .format(100*pred_xgb_valid.sum()/len(data_valid)))
        pred_xgb_proba_valid = xgb_model.predict_proba(data_valid[features])[:,1]
    print("Train set scores:\n",evalscores(y_train,pred_xgb_train,pred_xgb_train_proba))
    print("Test set scores:\n",evalscores(y_test,pred_xgb_test,pred_xgb_test_proba))

#%%# PERFORMING RANDOMIZED SEARCH ON XGBOOST
clf = xgb.XGBClassifier(nthread=16, learning_rate=0.01)

# Utility function to report best scores
def reporting(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# specify parameters and distributions to sample from
param_dist = {"max_depth": [1, 2, 3, 4, 5, 10],
              "min_child_weight":[1,2,3],
              "n_estimators": [200, 300, 400,500],
              "reg_alpha": [.005,.004,0.001, 0.01,.1, .2, .3, .4, .5, .6,1,],
              "reg_lambda": [.1,.2,.3,.01,.05,.001,.005,.0001,.0005],
              "subsample": [.9, 1],
              "gamma": [0, 1, 2, 3, 4, 5, 6, 8],
              "scale_pos_weight": [15,15.2,15.5,15.6,15.7,16,16.2]
              }

from sklearn.metrics import make_scorer
scorer = make_scorer(matthews_corrcoef, needs_proba=True)

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   scoring=scorer)

start = time.time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
reporting(random_search.cv_results_)

pred_random_train = random_search.predict(X_train)
pred_random_test = random_search.predict(X_test)
train_score = matthews_corrcoef(y_train,pred_random_train)
test_score = matthews_corrcoef(y_test,pred_random_test)
#train_score = f1_score(y_train,pred_random_train,average='micro')
#test_score = f1_score(y_test,pred_random_test,average='micro')
print("Train score: {0:.4f} - Test score: {1:.4f}".format(train_score, test_score))

#%% TENSORFLOW
# If first layer has less nodes than features then --> bad predictions
#Manual class balancing
def classbalance(dat, thresh):
    print("There are {:.2f}% of positive predictions in the original data set" 
          .format(100*dat['target'].sum()/len(dat)))
    
    #Fetch only positive entries
    df_positives_train_data = dat.loc[dat['target']==1]
    proportion = 0
    while proportion < thresh:
        #Percentage of positive target
        proportion = 100*dat['target'].sum()/len(dat)
        #Duplicate to balance data
        dat = dat.append(df_positives_train_data)
        print("There are {:.2f}% of positive predictions in the train set"
              .format(proportion))
        
    return dat

data_new_balanced = classbalance(data_new.copy(),22)

train, test = train_test_split(data_new_balanced, train_size=0.85,\
                               test_size=0.15)
#train, test = train_test_split(data_new, train_size=0.80,\
#                               test_size=0.20)
#train_balanced = classbalance(train,20)
#
#X_train, y_train, X_test, y_test = \
#    train_balanced[features], train_balanced[target], test[features], test[target]
X_train, y_train, X_test, y_test = \
    train[features], train[target], test[features], test[target]
print("There are {:.2f}% of positive predictions in the train set"
      .format(100*y_train.sum()/len(y_train)))
print("There are {:.2f}% of positive predictions in the test set"
      .format(100*y_test.sum()/len(y_test)))


def testinput(X_test, y_test):
    """
    Creates the test inputs
    """
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(X_test)},
            y=np.array(y_test),
            num_epochs=1,
            shuffle=False)
    return test_input_fn

def dnn_model(hu, model_dir, features, beta1=0.8, beta2=0.4):
    # Specify the shape of the features columns
    feature_columns = [tf.feature_column.numeric_column("x", 
                                                        shape=[len(features),1]
                                                        )]
    # Build n layer DNN with hu units (hu is an array)
    # The default optimizer is "AdaGrad" but we can specify another model
    classifier = tf.estimator.DNNClassifier\
                    (feature_columns=feature_columns,
                     hidden_units=hu,
                     n_classes=2,                                          
                     model_dir=model_dir,
                     optimizer=tf.train.ProximalAdagradOptimizer\
                                 (learning_rate=1e-3,
                                  l1_regularization_strength=0.0,
                                  l2_regularization_strength=1e-4 #ridge
                                 )
                    )
    '''
    OTHER OPTIMIZER
    optimizer=tf.train.AdamOptimizer(
                                                learning_rate=0.00001,
                                                beta1=0.1,
                                                beta2=0.99),
    '''
    
# Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train)},
        y=np.array(y_train.values.reshape((len(y_train),1))),
        num_epochs=None,
        shuffle=True)
    return classifier, train_input_fn

t_init=time.time()
# 3-layers
classifier, train_input_fn = dnn_model([228,600,600,600], "./tmp/DNN1", features)
#Let's train
classifier.train(input_fn=train_input_fn, max_steps=20000)
#classifier.evaluate(input_fn=train_input_fn, steps=10000)
test_input_fn = testinput(X_test,y_test)  

validation_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(data_valid_new[features])}, 
        y=None,
        num_epochs=1,
        shuffle=False)

training_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(X_train[features])}, 
        y=None,
        num_epochs=1,
        shuffle=False)

print("Predicting on train set...")
pred_tf_train_data = classifier.predict(input_fn=training_input_fn)
print("Predicting on test set...")    
pred_tf_test_data_temp = classifier.predict(input_fn=test_input_fn)
print("Predicting on validation set...")
pred_valid_temp = classifier.predict(input_fn=validation_input_fn)

print("Working on train predictions...")
predictions_tf_train = list(pred_tf_train_data)
pred_tf_train = list()
pred_tf_train_proba = list()
for i,p in enumerate(predictions_tf_train):
    pred_tf_train.append(p['class_ids'][0])
    pred_tf_train_proba.append(p['probabilities'][1])

print("Working on validation predictions...")    
predictions = list(pred_valid_temp)
pred_tf_validation = list()
pred_tf_validation_proba = list()
for i,p in enumerate(predictions):
    pred_tf_validation.append(p['class_ids'][0])
    pred_tf_validation_proba.append(p['probabilities'][1])

print("Working on test predictions...")
pred_tf_test_temp = list(pred_tf_test_data_temp)
pred_tf_test_proba = list()
pred_tf_test = list()
for i,p in enumerate(pred_tf_test_temp):
    pred_tf_test_proba.append(p['probabilities'][1])
    pred_tf_test.append(p['class_ids'][0])
print("Total processing time: {0:.3f}".format(time.time()-t_init))
print("Computing roc auc curve...")   
fpr, tpr, _ = roc_curve(y_test, pred_tf_test_proba,pos_label=1)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')

print("There are {:.2f}% of positive predictions on the train set"
      .format(100*pred_tf_train.count(1)/len(y_train)))
print("There are {:.2f}% of positive predictions on the test set"
      .format(100*pred_tf_test.count(1)/len(y_test)))
print("Number of positives in test set: ",y_test.sum())
print("There are {:.2f}% of positive predictions on the validation set"
      .format(100*pred_tf_validation.count(1)/len(data_valid)))

pred_tf_test = np.array(pred_tf_test)
pred_tf_train = np.array(pred_tf_train)

print("Train set scores:\n",evalscores(y_train,pred_tf_train,pred_tf_train_proba))
print("Test set scores:\n",evalscores(y_test,pred_tf_test,pred_tf_test_proba))


#%%
submit(pred_tf_validation_proba,data_valid_new,"file.csv")


#%% PLOT
plt.figure()
f, axes = plt.subplots(2, 1)
axes[0].plot(components,scores_train, components,scores_test)
axes[0].set_ylabel('Gini index')
axes[0].legend(["Train", "Test"])
axes[1].plot(components, times)
axes[1].set_ylabel('Fitting time (s)')
axes[1].set_xlabel("Number of components")