import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation, GlobalAveragePooling1D
from keras.layers import BatchNormalization, GlobalAveragePooling1D
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from livelossplot import PlotLossesKeras
from keract import get_activations, display_activations

from ecgutils import plot_confusion_mat, smooth, plot_saliency, plot_cam_rhythm_2
from ecgutils import get_n, plot_cam_background

from vis.visualization import visualize_saliency, visualize_cam, visualize_activation
from vis.utils import utils
from keras import activations

#%% Variables definition
SEED = 7
WDIR = "C://Users//amont//Desktop//Thesis//"

#%% Data Loading
glostrup = pd.read_csv(WDIR + 'glostrup_targets.csv')

data = pd.read_csv(WDIR + "data/rhythm_V5.csv")
rhythm = data.values

#Target variable
y = glostrup.sbrad
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
#model_sbrad.add(GlobalAveragePooling1D())
         

#model_sbrad.add(Flatten())
model_sbrad.add(GlobalAveragePooling1D())
#model_sbrad.add(Dense(128, activation = 'relu'))
#model_sbrad.add(Dropout(0.2))

model_sbrad.add(Dense(2, activation='sigmoid'))
# %% Model specifics
weights = {True: 3,
           False: 1}

model_sbrad.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0012,),
              metrics=['accuracy'])

#%% Trainig

history = model_sbrad.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=100, 
                    validation_data=(x_test, y_test),
                    callbacks=[PlotLossesKeras()],
                    verbose=1,
                    class_weight=weights)

#%% Model save/load

model_sbrad = load_model(WDIR + "source/ISCE/models/sbrad_cnn_rhythm.hdf5")
#model_sbrad.save(WDIR + "source/src/mdl/sbrad_GAP.hdf5")

#%% Model evaluation

y_pred = model_sbrad.predict(x_test)
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

y_pred_sbrad = model_sbrad.predict(x_test).ravel()
fpr_sbrad, tpr_sbrad, thresholds_sbrad = roc_curve(y_true, y_pred)
auc_sbrad = auc(fpr_sbrad, tpr_sbrad)

#%% Benchmark: smv with gaussian kernel, rf, logistic, decision tree, knn 

#%% Saliency visualization
layer_idx = -1
model_sbrad.layers[layer_idx].activation = activations.linear
model_sbrad = utils.apply_modifications(model_sbrad)
#%%
n = get_n(testid=4675, df=glostrup, index=test_index)
example = (x_test[n, :, :]).astype(dtype=float)
#%% Class activation Map 
cam = visualize_cam(model_sbrad, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


#plot_cam_rhythm_2(example,cam)
plot_cam_background(example, smooth(cam,30), glostrup, n, "sbrad", y_test[n])
#%%

a = np.where(np.argmax(y_test,1)==1)[0]
a = a[1:12]
for n in a:
    example = (x_test[n, :, :]).astype(dtype=float)
    cam = visualize_cam(model_sbrad, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


    plot_cam_background(example,cam, glostrup, n, "sbrad", y_test[n])
    
    #plt.savefig('C://Users//amont//Desktop//Thesis//attention_plot/'+str(n)+'.png')    
    
#%% Reset session
from keras import backend as K
import tensorflow as tf
curr_session = tf.get_default_session()
# close current session
if curr_session is not None:
    curr_session.close()
# reset graph
K.clear_session()
# create new session
s = tf.InteractiveSession()
K.set_session(s)    