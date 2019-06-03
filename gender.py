import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import regularizers
import keras.backend as K
from keras.models import load_model
import keras.activations


from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve

from livelossplot import PlotLossesKeras
from keract import get_activations, display_activations

from ecgutils import plot_confusion_mat, smooth, plot_saliency, plot_saliency_median, plot_cam_median, get_n

from vis.visualization import visualize_saliency, visualize_cam
from vis.utils import utils
from keras import activations

#%% Variables definition
SEED = 7
WDIR = "C://Users//amont//Desktop//Thesis//"

#%% Data Loading
glostrup = pd.read_csv(WDIR + 'glostrup_targets.csv')

data = pd.read_csv(WDIR + "data/median_V5.csv")
median = data.values

#Target variable
y = glostrup.sexnumeric
y = to_categorical(y)

# Training (67%) and test (33%) set. Manually defined to keep track of the IDs
test_index = np.random.choice(range(6667), size=2201, replace=False)
train_index = np.setdiff1d(range(6667), test_index)

x_train = median[train_index, :]
x_train = np.expand_dims(x_train, axis=2)

x_test = median[test_index, :]
x_test = np.expand_dims(x_test, axis=2)

y_train = y[train_index, :]
y_test = y[test_index, :]

#%% Define the network architecture
model = Sequential()

model.add(Conv1D(filters=128,
                 kernel_size=4,
                 input_shape=x_train[0].shape,
                dilation_rate=2))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters=64,
                 kernel_size=3,
                dilation_rate=4))
model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32,
                 kernel_size=8,
                dilation_rate=3))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=16,
                 padding='valid',
                 kernel_size=3,
                dilation_rate=3))
model.add(BatchNormalization())
model.add(Activation('relu'))


#model.add(Conv1D(filters=32,
#                 padding='valid',
#                 kernel_size=3))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#
#model.add(MaxPooling1D(2))
#
#model.add(Conv1D(filters=32,
#                 padding='valid',
#                 kernel_size=3))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

          

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))
# %% Model specifics

weights = {True: 1,
           False: 1}

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])

#%% Trainig

history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=30, 
                    validation_data=(x_test, y_test),
                    callbacks=[PlotLossesKeras()],
                    verbose=1,
                    class_weight=weights)

#%% Model evaluation
model_sex = load_model(WDIR + "source/ISCE/models/gender_cnn.hdf5")
#model.save(WDIR + "source/src/mdl/gender1.hdf5")
y_pred = model_sex.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

confusion = confusion_matrix(y_true, y_pred)

test_acc = accuracy_score(y_true, y_pred)
bal_test_acc = balanced_accuracy_score(y_true, y_pred)

print('In the test set, there are %d positive, %d negative' % (sum(y_true), len(y_true) - sum(y_true)))

print("The test accuracy is", test_acc)
print("The balanced test accuracy is", bal_test_acc)   

plot_confusion_mat(confusion, ["False", "True"],cmap="Reds", normalize=False)


#%% Misclassified analysis
misclassified = y_true - y_pred

false_positive = glostrup.loc[test_index[np.where(misclassified ==  -1)]]
false_negative = glostrup.loc[test_index[np.where(misclassified ==  1)]]
#%% Evaluation of the model
f1_score = f1_score(y_true, y_pred)

y_pred_sex = model_sex.predict(x_test).ravel()
fpr_sex, tpr_sex, thresholds_sex = roc_curve(y_true, y_pred)
auc_sex = auc(fpr_sex, tpr_sex)

#%% Benchmark: smv with gaussian kernel, rf, logistic, decision tree, knn 

#%% Saliency visualization
layer_idx = -1
model_sex.layers[layer_idx].activation = activations.linear
model_sex = utils.apply_modifications(model_sex)
#%%
#n = get_n(testid=1012, df=glostrup, index=test_index)
example = (x_test[15, :, :]).astype(dtype=float)
#%% Class activation Map 
from ecgutils import plot_cam_background_median
cam = visualize_cam(model_sex, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


#plot_cam_rhythm_2(example,cam)
plot_cam_background_median(example, cam, glostrup, n, "gender", y_test[n])