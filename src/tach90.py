import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#plt.style.use("bmh")
#get_ipython().run_line_magic('matplotlib', 'inline')

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
from sklearn.metrics import accuracy_score

from livelossplot import PlotLossesKeras
from keract import get_activations, display_activations

from ecgutils import plot_confusion_mat, smooth, plot_saliency, plot_cam_rhythm_2

from vis.visualization import visualize_saliency, visualize_cam
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
model_stach = Sequential()

model_stach.add(Conv1D(filters=64,
                 kernel_size=10,
                 input_shape=x_train[0].shape,
                dilation_rate=7))
model_stach.add(BatchNormalization())
model_stach.add(Activation('relu'))

model_stach.add(Conv1D(filters=32,
                 kernel_size=7,
                dilation_rate=7))
#model_stach.add(BatchNormalization())
model_stach.add(Activation('relu'))

model_stach.add(MaxPooling1D(pool_size=2))

model_stach.add(Conv1D(filters=32,
                 padding='valid',
                 kernel_size=5,
                dilation_rate=3))
#model_stach.add(BatchNormalization())
model_stach.add(Activation('relu'))

model_stach.add(Conv1D(filters=32,
                 padding='valid',
                 kernel_size=22))
#model_stach.add(BatchNormalization())
model_stach.add(Activation('relu'))

model_stach.add(MaxPooling1D(2))

model_stach.add(Conv1D(filters=32,
                 padding='valid',
                 kernel_size=20))
#model_stach.add(BatchNormalization())
model_stach.add(Activation('relu'))

          

model_stach.add(Flatten())

model_stach.add(Dense(128, activation = 'relu'))
#model_stach.add(Dropout(0.2))

model_stach.add(Dense(2, activation='sigmoid'))
# %% Model specifics
weights = {True: 3,
           False: 1}

model_stach.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0012),
              metrics=['accuracy'])

#%% Trainig

history = model_stach.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=100, 
                    validation_data=(x_test, y_test),
                    callbacks=[PlotLossesKeras()],
                    verbose=1,
                    class_weight=weights)

#%% Model save/load

model_stach = load_model(WDIR + "source/ISCE/models/tach90_cnn_rhythm.hdf5")
#model_stach.save(WDIR + "source/src/mdl/tach90.hdf5")

#%% Model evaluation

y_pred = model_stach.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

test_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred)
bal_test_acc = balanced_accuracy_score(np.argmax(y_test, axis=1), y_pred)

print('In the test set, there are %d positive, %d negative' % (sum(np.argmax(y_test, 1)), len(np.argmax(y_test, 1)) - sum(np.argmax(y_test, 1))))

print("The test accuracy is", test_acc)
print("The balanced test accuracy is", bal_test_acc)   

plot_confusion_mat(confusion, ["False", "True"],cmap="Reds", normalize=False)


#%% Misclassified analysis

misclassified = np.argmax(y_test, 1) - y_pred

false_positive = glostrup.loc[test_index[np.where(misclassified ==  -1)]]
false_negative = glostrup.loc[test_index[np.where(misclassified ==  1)]]
#%%


def get_n(testid):
    n = np.where(glostrup.testid == testid)
    return n




#%% Saliency visualization
layer_idx = -1
model_stach.layers[layer_idx].activation = activations.linear
model_stach = utils.apply_modifications(model_stach)
#%%
n = a[20]
example = (x_test[n, :, :]).astype(dtype=float)
print("y_test", y_test[n])
print("y_pred", y_test[n])

#%% Class activation Map 
from ecgutils import plot_cam_rhythm_2, plot_cam_background
cam = visualize_cam(model_stach, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


plot_cam_rhythm_2(example,cam)

plot_cam_background(example, cam)
#%%
grads = visualize_saliency(model_stach, layer_idx, filter_indices=1, seed_input=example, backprop_modifier='guided')

fig = plt.figure(figsize=(14,7))
plt.plot(grads)
plt.show()

plot_saliency(example, grads)



#%%

a = np.where(np.argmax(y_test,1)==1)[0]
a = a[0:25]
for n in a:
    example = (x_test[n, :, :]).astype(dtype=float)
    cam = visualize_cam(model_stach, layer_idx, filter_indices=1, seed_input=example, 
                    penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)


    plot_cam_rhythm_2(example,cam)
    
    plt.savefig('C://Users//amont//Desktop//Thesis//attention_plot/'+str(n)+'.png')    