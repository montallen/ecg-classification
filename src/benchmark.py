import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate

import time
import warnings
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
#
warnings.filterwarnings('ignore')
WDIR = "C://Users//amont//Desktop//Thesis//"
SEED=7

glostrup = pd.read_csv(WDIR + 'glostrup_targets.csv', sep = ',').dropna()

#X_clf = glostrup[glostrup.columns.difference(['testid','sex','sexnumeric'])]
X_clf = glostrup[["sexnumeric", "bmi","qrs","qt","pr"]]#,"p_peak_amp_v5","q_peak_amp_v5","r_peak_amp_v5", "s_peak_amp_v5", "t_peak_amp_v5", "rr"]]
y_clf = glostrup.sbrad

#sokolow_lyon = glostrup.s_peak_amp_v1.values + np.maximum(glostrup.r_peak_amp_v5.values,
#                                                          glostrup.r_peak_amp_v6.values)
#
#y_clf = sokolow_lyon>3500

models_clf = [LogisticRegression(),
          RidgeClassifier(alpha=30), 
          GaussianNB(), 
          RandomForestClassifier(), 
          MLPClassifier(), 
          DummyClassifier(strategy="most_frequent")]

names_clf = ["Logistic Regression", 
         "Ridge Regression", 
         "Naive Bayes",
         "Random Forest",
         "Multi-Layer Perceptron", 
         "Most Frequent"]

scoring_clf = {'acc': 'accuracy',
               'f1_macro': 'f1_macro',
              'precision_macro': 'precision_macro',
              'recall_macro':'recall_macro'}
results_clf = []

for model, name in zip(models_clf, names_clf):
    scores_clf = cross_validate(model, 
                                X_clf, y_clf, 
                                scoring=scoring_clf, 
                                cv=10, 
                                return_train_score=True)
    
    results_clf.append(scores_clf)
    
    msg = "%s: Accuracy: %f (%f); F1 score: %f; Precision: %f; Recall: %f." % (name, 
                                         scores_clf["test_acc"].mean(), 
                                         scores_clf["test_acc"].std(), 
                                         scores_clf["test_f1_macro"].mean(),
                                         scores_clf['test_precision_macro'].mean(),
                                         scores_clf['test_recall_macro'].mean())
    print(msg)
    print('\n')