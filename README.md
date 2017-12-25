# Welcome to my machine-learning repo

## Kaggle's Porto Seguro safe driver competition:
* [Complete model](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Porto%20Seguro/ExtraTress_PCA_DNN_Logit.py)

Solution to this classification problem using different techniques : logistic regression, XGBoost classifier and DNN (tensorflow). Features selection is performed with ExtraTreesClassifier, and PCA can be used if the data is binarized (leading to too many features). 

Data preprocessing is done with a [specific class](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Porto%20Seguro/encoder.py).


## Kaggle's Titanic competition:
* [Random forest using GraphLab](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Titanic/Random-forest.ipynb)

* [Basic decision tree using GraphLab](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Titanic/Simple_decision_tree.ipynb)

* [Home-made k-NN model](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Titanic/k-NN.ipynb)

* [Multiple models with scikit-learn](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/Titanic/Titanic-Multi-model.py) 

This model combines algorithms such as : RandomForestClassifier, AdaBoosClassifier, DecisionTreeClassifier, SVM, KNN. For the final prediction EnsembleVotingClassifier is used. Best score is **0.79904**

## Kaggle's House price competition:

* [Jupyter notebook, multiple models with scikit-learn](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/House%20Prices/HousePrice-Notebook_for_kaggle.ipynb)

This model combines several regression techniques LASSO, Elastic Net, Gradient Boosting, AdaBoost, XGBoost. Best score is **~ 0.1250**

* [Using MLBox](https://github.com/DarquesM/Machine-Learning/blob/master/Kaggle/House%20Prices/HousePrices-MLBox.py)

[MLBox](https://github.com/AxeldeRomblay/MLBox) is a python automated machine-learning library, written by Axel de Romblay

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
