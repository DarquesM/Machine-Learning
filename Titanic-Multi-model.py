# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:17:23 2017

@author: Michael
"""
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, classification
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.svm import SVC
#import timeit
import time

#%% FONCTIONS

def plot_cat(data, x_axis, y_axis, hue):
    plt.figure()    
    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=data)
    sns.set_context("notebook", font_scale=1.6)
    plt.legend(loc="upper right", fontsize="medium")    
    
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def median_age(data,Pclass):
    med_age = round(data["Age"][data["Pclass"]==Pclass].median())
    return med_age

def rfc(n_est,msl,mis):
    model = sklearn.ensemble.RandomForestClassifier(n_estimators= n_est,min_samples_leaf=msl,min_impurity_split=mis)
    #msl = 4, n_est = 58, mis = 5e-8
    return model

def cross_val(model,data,features):
    scores = (cross_val_score(model,data[features],data["Survived"],cv=10,n_jobs=1)).mean()
    return scores

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")    
#%% Importation des données
data_validation = pd.read_csv("test.csv")
data = pd.read_csv("train.csv")

#%%
print("Regardons les données manquantes")
pd.isnull(data).sum()
#%%
print("Combien avons nous de morts et de survivants ?")
Survived_counts = data.Survived.value_counts()
print(Survived_counts)
#%%
#Plots par catégorie    
plot_cat(data,"Pclass", "Survived", "Sex") 
plot_cat(data,"Sex", "Survived", None) 
plot_cat(data,"Pclass", "Survived", None)
plot_correlation_map(data)

#%%
''' *** EXPLORATION STATISTIQUE *** '''
#Les 4 lignes ci-dessous sont équivalentes. Je compte le nombre de morts/vivants par sexe
#women = [len(data.loc[(data.Sex == 0)&(data.Survived==0)]),len(data.loc[(data.Sex == 0)&(data.Survived==1)])]
#men = [len(data.loc[(data.Sex == 1)&(data.Survived==0)]),len(data.loc[(data.Sex == 1)&(data.Survived==1)])]
women = data.loc[(data.Sex == 0)].Survived.value_counts(sort=False)
men = data.loc[(data.Sex == 1)].Survived.value_counts(sort=False)
sp.stats.chi2_contingency([women,men])
#p-value << 1 donc l'hypothèse est gardée/rejetée (check) et donc on a bien une différence entre homme/femmes pour Survived
CabineFalse = data.loc[(data.Cabin == 0)].Survived.value_counts(sort=False)
CabineTrue = data.loc[(data.Cabin == 1)].Survived.value_counts(sort=False)
sp.stats.chi2_contingency([CabineFalse,CabineTrue])                                                  
#%%
#The missing "Age" values are replaced by the median age for each Class
data.loc[(data.Age.isnull()) & (data["Fare"] == 0), "Age"] = 0
for i in [1,2,3]:
    data.loc[(data.Age.isnull()) & (data["Pclass"] == i), "Age"] = median_age(data,i)
data.loc[(data.Sex == "male", "Sex")]=1
data.loc[(data.Sex == "female", "Sex")]=0

data.loc[(data.Cabin.isnull()==False), "Cabin"] = 1
data.loc[(data.Cabin.isnull()), "Cabin"] = 0
data["Agebin"] = pd.cut(data["Age"],bins=[0,4,12,25,60,85], labels=[1,2,3,4,5])
data["Famille"] = 1+data["SibSp"]+data["Parch"]
data.describe()
#%%
#On applique les mêmes transformations au validation set
#The missing "Age" values are replaced by the median age for each Class
data_validation.loc[(data_validation.Age.isnull()) & (data_validation["Fare"] == 0), "Age"] = 0
for i in [1,2,3]:
    data_validation.loc[(data_validation.Age.isnull()) & (data_validation["Pclass"] == i), "Age"] = median_age(data,i)
data_validation.loc[(data_validation.Sex == "male", "Sex")]=1
data_validation.loc[(data_validation.Sex == "female", "Sex")]=0

data_validation.loc[(data_validation.Cabin.isnull()==False), "Cabin"] = 1
data_validation.loc[(data_validation.Cabin.isnull()), "Cabin"] = 0

data_validation["Agebin"] = pd.cut(data_validation["Age"],bins=[0,4,12,25,60,85], labels=[1,2,3,4,5])
data_validation["Famille"] = 1+data_validation["SibSp"]+data_validation["Parch"]
data_validation.loc[(data_validation.Fare.isnull()), "Fare"] = 9
data_validation.describe()

#%%
twoway = pd.crosstab(data['Survived'], data["Agebin"], margins=True)
twoway
from statsmodels.graphics.mosaicplot import mosaic
mosaic(data,['Agebin','Survived'])
twoway[1][0] #Nombre de bébés ayant coulé


#%%
target = "Survived"
features = ["Pclass", "Sex","Age", "Cabin","Famille"]
train,test = train_test_split(data,test_size=0.2)

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]
X_valid = data_validation[features]
X = data[features]
y = data[target]

#%% k-NN MODEL
''' *** TRAITEMENT ET NORMALISATION DES DONNEES POUR LE MODELE K-NN ***'''
x = data.values #Get only values from the dataframe
min_max_scaler = preprocessing.MinMaxScaler() #Create the scaling function
#x_scaled = min_max_scaler.fit_transform(x)
data["Age_n"] = min_max_scaler.fit_transform(data["Age"].reshape(-1,1))
data["Pclass_n"] = min_max_scaler.fit_transform(data["Pclass"].reshape(-1,1))
data["Famille_n"] = min_max_scaler.fit_transform(data["Famille"].reshape(-1,1))
data["Fare_n"] = min_max_scaler.fit_transform(data["Fare"].reshape(-1,1))
y = data_validation.values #Get only values from the dataframe
min_max_scaler = preprocessing.MinMaxScaler() #Create the scaling function
data_validation["Age_n"] = min_max_scaler.fit_transform(data_validation["Age"].reshape(-1,1))
data_validation["Pclass_n"] = min_max_scaler.fit_transform(data_validation["Pclass"].reshape(-1,1))
data_validation["Famille_n"] = min_max_scaler.fit_transform(data_validation["Famille"].reshape(-1,1))
data_validation["Fare_n"] = min_max_scaler.fit_transform(data_validation["Fare"].reshape(-1,1))
#data_validation.loc[(data_validation.Sex == "female", "Sex")]=0
features = ["Pclass_n", "Sex","Famille_n", "Age_n", "Fare_n", "Cabin"]
train,test = train_test_split(data,test_size=0.2)
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]
X_valid = data_validation[features]
X = data[features]
y = data[target]

#%%

k = 10
model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,n_jobs=1)
#model_knn.fit(X_train,y_train)
model_knn.fit(X,y)

distances, idx = model_knn.kneighbors() #Je récupère les k plus proches voisins pour chaque point
#data["Survived"][idx[0]]
predictions_kNN = model_knn.predict(X_valid)

''' J'adapte le modèle pour régler le seuil de prédiction. Je récupère la probabilité
et j'affecte les valeurs survived=0 ou 1, pour différentes valeurs du seuil (moins de 0.5 marche mieux)
 '''

#predictions_proba = model_knn.predict_proba(X_valid)
#predictions = pd.DataFrame(columns=["Survived"])
#predictions["Survived"] = [i for i in range(len(X_valid))]
#predictions.loc[predictions_proba[:,0]>=0, "Survived"]=0
#for i in range(len(X_valid)):
#    if predictions_proba[i,0]>=0.4:
#        predictions["Survived"][i] = 0
#    else:
#        predictions["Survived"][i] = 1

#%%
model_rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=55,min_samples_leaf = 2, min_samples_split=3, min_impurity_split=6.5e-8)
model_rfc.fit(X_train,y_train)
predictions_rfc = model_rfc.predict(X_test)
#predictions_final = model.predict(X_valid)
print("Training accuracy : ", model_rfc.score(X_train,y_train))
print("Test accuracy : ", model_rfc.score(X_test,y_test))

#%% *** SVM - SVC ***
model_svc = SVC(probability=True)
model_svc.fit(X_train,y_train)
predict_svc = model_svc.predict(X_test)


#%% *** RANDOM FOREST ***
model_rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=55,min_samples_leaf = 2, min_samples_split=3, min_impurity_split=6.5e-8)
scores_test = cross_val_score(model_rfc,data[features],data[target],cv=10) #Donne le score sur le test après avoir procédé à un fit
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2))
model_rfc.fit(X_test,y_test)
print("Features importance")
for feat,i in zip(features,[j for j in range(len(features))]):
    print(feat,model_rfc.feature_importances_[i])

#%%GridSearchCV for RANDOM FOREST
t_init = time.time()

rfc_model = sklearn.ensemble.RandomForestClassifier()

param_grid = { 
    'n_estimators': [51,52,53,54,55,56,57,58,59,60],
    'min_samples_leaf' : [1,2,3],
    'min_samples_split' : [2,3,4],
    'min_impurity_split': [6.5e-8,7e-8,7.5e-8,8e-8,9e-8,1e-7]
}
CV_rfc = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv= 5,n_jobs=4)
CV_rfc.fit(X, y)
print(CV_rfc.best_params_)
t_final = time.time()
print("Temps d'exécution de la routine: ", t_final-t_init)

#%% *** DECISION TREE ***
model_dtc = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_split=4e-8,min_samples_leaf=1, min_samples_split=5, splitter='random')
scores_test = cross_val_score(model_dtc,data[features],data[target],cv=10) #Donne le score sur le test après avoir procédé à un fit
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2))
model_dtc.fit(X_test,y_test)
print("Features importance")
for feat,i in zip(features,[j for j in range(len(features))]):
    print(feat,model_dtc.feature_importances_[i])

#%%GridSearchCV for DECISION TREE
t_init = time.time()

dt_model = sklearn.tree.DecisionTreeClassifier()

param_grid = { 
    'criterion' : ["gini","entropy"],
    'splitter' : ["best", "random"],
    'max_depth': [5,6,7,8,9],
    'min_samples_leaf' : [1,2],
    'min_samples_split' : [2,3,4,5],
    'min_impurity_split': [5e-9,1e-8,2e-8,3e-8,4e-8,5e-8,6e-8,7e-8,1e-7]
}
CV_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv= 5,n_jobs=4)
CV_dt.fit(X, y)
print(CV_dt.best_params_)
t_final = time.time()
print("Temps d'exécution de la routine: ", t_final-t_init)



#%%
''' *** ENSEMBLE VOTING CLASSIFIER *** '''
#
#model_knn
#model_rfc
#model_svc
#clf1 = sklearn.neighbors.KNeighborsClassifier()
#clf2 = sklearn.ensemble.RandomForestClassifier
#clf3 = sklearn.svm.SVC()
#Ensemble Classifier
eclf = VotingClassifier(estimators=[('knn',model_knn)
    ,('rfc',model_rfc),('svc',model_svc)],voting='soft',
    weights=[1,1,1])

for clf,label in zip([model_knn,model_rfc,model_svc,eclf],
                     ['kNN Model','RandomForest', "SVC","Ensemble"]):
    scores = cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))   


eclf.fit(X_test,y_test)

predictions_eclf = eclf.predict(X_valid)

#Amélioration du score de 0.78947 à 0.79904 avec 3 modèles dans le eclf


#%%
''' *** EXTRACTION DES RESULTATS POUR SOUMISSION *** '''
#predictions = CV_rfc.predict(X_valid)
#predictions = model.predict(X_valid)
#predictions = model_knn.predict(X_valid)
result = pd.DataFrame(columns=["PassengerId", "Survived"])
result["PassengerId"] = data_validation['PassengerId']
result["Survived"] = predictions_eclf
result.to_csv("Submission-eclf6.csv", index=False) #On n'oublie pas d'enlever l'index

#k-NN basique donne 0.67464 (5 voisins et j'applique predict sur X sans paramétrer)
#k-NN 10 voisins avec threshold = 0.4 donne 0.76555

#%%

''' 
CI-DESSOUS CELLULES POUR TESTER INDIVIDUELLEMENT L'INFLUENCE DES PARAMETRES SUR LE SCORE
'''


#%%Min impurity split
#%% Evaluation du modèle
#Exécuter seulement une seule cellule
train_score = np.array([])
test_score = np.array([])

mis = np.array([1e-8,5e-8,6e-8,7e-8,8e-8,9e-8,1e-7,2e-7,3e-7,4e-7,5e-7,6e-7,7e-7,8e-7,9e-7,1e-6])
t_init = time.time()
n_est = 58
msl = 5
for m in mis:
    model = rfc(n_est,msl,m)
    test_score = np.append(test_score,cross_val(model,data,features))
    model.fit(X_train,y_train)
    train_score = np.append(train_score,model.score(X_train,y_train))
t_final = time.time()
print("Temps d'exécution de la routine: ", t_final-t_init)
plt.figure()    
plt.plot(mis,train_score, label ="Training accuracy")
plt.plot(mis,test_score, label ="Test accuracy")
plt.xlabel("Min impurity split")
plt.ylabel("Score")
plt.legend(fontsize="medium") 
#%% Min_sample leaf
msl = np.arange(1,15,1)
t_init = time.time()
n_est = 58
for m in msl:
    model = rfc(n_est,m)
    test_score = np.append(test_score,cross_val(model,data,features))
    model.fit(X_train,y_train)
    train_score = np.append(train_score,model.score(X_train,y_train))
t_final = time.time()
print("Temps d'exécution de la routine: ", t_final-t_init)

#%% Nombre estimators
t_init = time.time()
n_est = np.arange(50,183,2)
for n in n_est:
    model = rfc(n)
    test_score = np.append(test_score,cross_val(model,data,features))
    model.fit(X_train,y_train)
    train_score = np.append(train_score,model.score(X_train,y_train))
    
t_final = time.time()
print("Temps d'exécution de la routine: ", t_final-t_init)

plt.figure()    
plt.plot(n_est,train_score, label ="Training accuracy")
plt.plot(n_est,test_score, label ="Test accuracy")
plt.xlabel("Number of estimators")
plt.ylabel("Score")
plt.legend(fontsize="medium")    

#%% Test plot avec légende

plt.figure()    
plt.plot(msl,train_score, label ="Training accuracy")
plt.plot(msl,test_score, label ="Test accuracy")
plt.xlabel("Min samples leaf")#
plt.ylabel("Score")
plt.legend(fontsize="medium") 

#%%
plt.figure()
ax = sns.violinplot(x=data["Age"],y=data["all"],hue=data["Survived"], split=True)
#ax = sns.stripplot(x="Age", y="all", data=data, hue = "Survived", jitter=True)
#plt.xlabel("Age")
plt.ylabel("")
#plt.legend(fontsize='large')
sns.set_context("notebook", font_scale=1.6)
#%%
plt.figure()
ax = sns.barplot(x="Cabin", y='Survived', hue="Pclass", data=data)
sns.set_context("notebook", font_scale=1.8)
plt.legend(loc="upper left",fontsize="small")   

ax.set_xticklabels(["Sans", "avec"])

