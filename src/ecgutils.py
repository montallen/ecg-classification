import matplotlib.pyplot as plt
import numpy as np



def plot_confusion_mat(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                    fontsize=20)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=20)


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("gender_mat.png", format='png', dpi=1000)

    plt.show()
    

def smooth(x,window_len=11,window='hanning'):

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y    


def plot_saliency(beispiel, grads):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = np.arange(0,5000)
    y = np.squeeze(beispiel[1:,:],1)
    dydx = grads
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(figsize=(14,7))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(4)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(beispiel[1:,:].min()-100, beispiel[1:,:].max()+100)
    plt.show()
    
    
def plot_saliency_median(beispiel, grads):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = np.arange(0,600)
    y = np.squeeze(beispiel[1:,:],1)
    dydx = grads
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(figsize=(14,7))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(3)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(beispiel[1:,:].min()-100, beispiel[1:,:].max()+100)
    plt.show()
    
    
def plot_cam_median(beispiel, cam):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = np.arange(0,600)
    y = np.squeeze(beispiel[1:,:],1)
    dydx = cam
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(figsize=(14,7))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(3)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(beispiel[1:,:].min()-100, beispiel[1:,:].max()+100)
    plt.show()
    
def plot_cam_rhythm(beispiel, grads):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = np.arange(0,5000)
    y = np.squeeze(beispiel[1:,:],1)
    dydx = grads
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(figsize=(14,7))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    
    
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(beispiel[1:,:].min()-100, beispiel[1:,:].max()+100)
    plt.show()
    
def plot_cam_rhythm_2(beispiel, grads):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    x = np.arange(0,5000)
    y = np.squeeze(beispiel[1:,:],1)
    dydx = grads
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    fig, axs = plt.subplots(2,1,figsize=(20,9),sharex=True, sharey=False)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[:])
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(beispiel[1:,:].min()-100, beispiel[1:,:].max()+100)
    axs[1].plot(np.arange(0,5000),dydx[1:])
    
    
    #plt.show()
def plot_cam_background(example, cam, df, n, target, y_true):
    x = np.arange(0,5001)    
    arrays = [cam.T for _ in range(1500)]
    a = np.stack(arrays, axis=0)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(a, cmap="autumn_r", origin='lower',extent=[x.min(),x.max(),-1000,2000])#example[1:].min(), example[1:].max()])
    plt.plot(example[1:], color='blue',lw=1)
    #testid = np.array(df.iloc[np.where(df.unnamed==n)].testid)[0]
    hr=int(np.array(df.iloc[np.where(df.unnamed==n)].hr))
    
    y_tr = "Positive" if (np.argmax(y_true) == 1) else "Negative"
    plt.colorbar()
    plt.title("Target: " + target + ". Ground truth: " + str(y_tr))
    
    #plt.show()

def get_n(testid,df, index):
    x = np.where(df.testid == testid)
    n = np.where(index==x)#ottieni un numero. Poi np.where(set_index==numero). Ottieni la n cercata
    return n[1][0]

def plot_cam_background_median(example, cam, df, n, target, y_true):
    x = np.arange(0,601)    
    arrays = [cam.T for _ in range(1500)]
    a = np.stack(arrays, axis=0)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(a, cmap="autumn_r", origin='lower',extent=[x.min(),x.max(),-500,1000])#example[1:].min(), example[1:].max()])
    plt.plot(example[1:], color='blue',lw=1)
    #testid = np.array(df.iloc[np.where(df.unnamed==n)].testid)[0]
    hr=int(np.array(df.iloc[np.where(df.unnamed==n)].hr))
    
    y_tr = "Positive" if (np.argmax(y_true) == 1) else "Negative"
    plt.colorbar()
    plt.title("Target: " + target + ". Ground truth: " + str(y_tr))