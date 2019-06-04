import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf

from keras import Sequential, activations
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import  Adam
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from livelossplot import PlotLossesKeras
from keract import get_activations, display_activations

from ecgutils import plot_confusion_mat, plot_cam_background, plot_roc
from ecgutils import get_n

from vis.visualization import visualize_cam
from vis.utils import utils


#%% Variables definition
warnings.filterwarnings('ignore')

SEED = 7
WDIR = "C://Users//amont//Desktop//Thesis//"
MODEL_LOAD = True
#%% Data Loading
glostrup = pd.read_csv(WDIR + 'glostrup_targets.csv')

data = pd.read_csv(WDIR + "data/rhythm_V5.csv")
rhythm = data.values

#Target variable
y = glostrup.hr>90
y = to_categorical(y)

# Training (67%) and test (33%) set. Manually defined to keep track of the IDs
test_index = np.random.choice(range(6667), size=2201, replace=False)
train_index = np.setdiff1d(range(6667), test_index)

x_train = rhythm[train_index, :]
x_train = np.expand_dims(x_train, axis=2)

x_test = rhythm[test_index, :]
x_test = np.expand_dims(x_test, axis=2)

y_train = y[train_index, :]
y_test = y[test_index, :]

#%% Define the network architecture
if MODEL_LOAD is False:
    model_sbrad = Sequential()

    model_sbrad.add(Conv1D(filters=64,
                     kernel_size=10,
                     input_shape=x_train[0].shape, 
                     activation = "relu"))
    
    model_sbrad.add(Conv1D(filters=32,
                     kernel_size=10,
                     dilation_rate=3,
                     activation='relu'))
    
    model_sbrad.add(Conv1D(filters=32,
                     padding='valid',
                     kernel_size=5,
                     dilation_rate=5,
                    activation="relu"))
    
    model_sbrad.add(Conv1D(filters=32,
                     padding='valid',
                     kernel_size=5,
                     dilation_rate=5,
                     activation="relu"))
    
    
    model_sbrad.add(Conv1D(filters=32,
                     padding='valid',
                     kernel_size=3,
                     dilation_rate=7, 
                     activation="relu"))     
    
    #model_sbrad.add(Flatten())
    model_sbrad.add(GlobalAveragePooling1D())
    #model_sbrad.add(Dense(128, activation = 'relu'))
    #model_sbrad.add(Dropout(0.2))
    
    model_sbrad.add(Dense(2, activation='sigmoid'))
#   

    weights = {True: 3,
               False: 1}
    
    model_sbrad.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0012,),
                  metrics=['accuracy'])

    history = model_sbrad.fit(x_train, y_train, 
                        batch_size=64, 
                        epochs=100, 
                        validation_data=(x_test, y_test),
                        callbacks=[PlotLossesKeras()],
                        verbose=1,
                        class_weight=weights)

#%% Model save/load

model_tach = load_model(WDIR + "source/src/mdl/tach90_cnn_rhythm.hdf5")
#model_sbrad.save(WDIR + "source/src/mdl/sbrad_GAP.hdf5")

#%% Model evaluation

y_pred = model_tach.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

confusion = confusion_matrix(y_true, y_pred)

test_acc = accuracy_score(y_true, y_pred)
bal_test_acc = balanced_accuracy_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)
print('In the test set, there are %d positive, %d negative' % (sum(y_true), 
                                                               len(y_true) - sum(y_true)))

print("The test accuracy is", test_acc)
print("The balanced test accuracy is", bal_test_acc)   
print("F1 score:", f1)
plot_confusion_mat(confusion, ["False", "True"],cmap="Reds", normalize=False)

#%% Evaluation of the model
y_pred_sbrad = model_tach.predict(x_test)

fpr_sbrad, tpr_sbrad, thresholds_sbrad = roc_curve(y_true, y_pred_sbrad[:,1])
auc_sbrad = auc(fpr_sbrad, tpr_sbrad)
plot_roc(fpr_sbrad, tpr_sbrad, auc_sbrad)

#%% Benchmark: Linear Regression, Random Forest, KNN, SVM, Most frequent class 
glostrup_baseline = pd.read_csv(WDIR + 'glostrup_targets.csv', sep = ',').dropna()

X_clf = glostrup_baseline[["sexnumeric", "bmi","qrs","qt","pr"]]
y_clf = glostrup_baseline.hr>90

models_clf = [LogisticRegression(),
          RandomForestClassifier(),
          KNeighborsClassifier(),
          SVC(),
          DummyClassifier(strategy="most_frequent")]

names_clf = ["LR", 
         "RF",
         "KNN",
         "SVM",
         "Most Frequent"]

scoring_clf = {'acc': 'accuracy',
               'f1_micro': 'f1_micro',
              'precision_micro': 'precision_micro',
              'recall_micro':'recall_micro', 
              'roc_auc': 'roc_auc'}
results_clf = []

for model, name in zip(models_clf, names_clf):
    scores_clf = cross_validate(model, 
                                X_clf, y_clf, 
                                scoring=scoring_clf, 
                                cv=10, 
                                return_train_score=True)
    
    results_clf.append(scores_clf)
    
    msg = "%s: Accuracy: %.3f (%.3f); F1 score: %.2f; Precision: %.2f; Recall: %.2f; AUC: %.2f" % (name, 
                                         scores_clf["test_acc"].mean(), 
                                         scores_clf["test_acc"].std(), 
                                         scores_clf["test_f1_micro"].mean(),
                                         scores_clf['test_precision_micro'].mean(),
                                         scores_clf['test_recall_micro'].mean(),
                                         scores_clf['test_roc_auc'].mean())
    print(msg)
    print('\n')

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
#%% Class activation map visualization
layer_idx = -1
model_tach.layers[layer_idx].activation = activations.linear
model_tach = utils.apply_modifications(model_tach)
#%% Get expample from test_id
n = get_n(testid=3, df=glostrup, index=train_index)

example = (x_train[n, :, :]).astype(dtype=float)
#%% Plot
cam = visualize_cam(model_tach, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


plot_cam_background(example, cam, glostrup, "Tachycardia", y_train[n])
#%% Misclassified analysis
misclassified = y_true - y_pred

false_positive = glostrup.loc[test_index[np.where(misclassified ==  -1)]]
false_negative = glostrup.loc[test_index[np.where(misclassified ==  1)]]    
#%% Reset session
curr_session = tf.get_default_session()
# close current session
if curr_session is not None:
    curr_session.close()
# reset graph
K.clear_session()
# create new session
s = tf.InteractiveSession()
K.set_session(s)    