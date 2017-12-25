# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:16:21 2017

@author: mdarq
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) #Je crée un meshgrid ave un pas de h entre le min et le max de x
    
    #xx.ravel() colle tous les bouts de xx les uns après les autres (on a donc un array)
    '''En gros l'opération à l'intérieur de la prédiction ci-dessous permet de passer d'une 
    forme A=[1,2,3][4,5,6] et B[7,8,9][10,11,12] à 
    array([[ 1,  7],
           [ 2,  8],
           [ 3,  9],
           [ 4, 10],
           [ 5, 11],
           [ 6, 12]])
    On commence par fusionner les arrays en [1,2,3,4,5,6] et [7,8,9,10,11,12]
    '''
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    '''
    Ci-dessous, ça ne marche pas car la prédiction prend les colonnes, il en faut 2
    Comme pour le train
    '''
    #clf.predict([xx.ravel(),yy.ravel()])
    #np.shape(n_row,n_col)    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape) # On passe de (61600) à (220,280)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()