# Lecture des données
import numpy as np
from sklearn.model_selection import train_test_split 

features = np.load('features.npy')
targets = np.load('targets.npy')
steroides = np.load('steroides.npy') #ajouter y control pour faire des courbes ROC
X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.4, random_state=42)

# Création du model
from keras.models import Sequential
from keras.layers import Dense, Convolution3D, Flatten, MaxPooling3D, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

model = Sequential() 

        # 1ère convolution 

model.add(Convolution3D(input_shape = (14,32,32,32), filters=64, kernel_size=5, padding='valid', data_format='channels_first'))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling3D(pool_size=(2,2,2), padding='valid', data_format='channels_first'))

        # 2ème convolution 

model.add(Convolution3D(filters=64, kernel_size=3, padding='valid', data_format='channels_first',))
model.add(LeakyReLU(alpha = 0.2))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
model.add(Dropout(0.4))

        # 3ème convolution 

model.add(Convolution3D(filters=128, kernel_size=2, padding='valid', data_format='channels_first',))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid', data_format='channels_first'))
model.add(Dropout(0.4))

        # Fully connected layer 3 to shape (3) for 3 classes

model.add(Flatten())
model.add(Dense(3))
model.add(Activation('softmax'))

## Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## Training
EarlyStopping(monitor='val_loss', min_delta=0.1, patience=6, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1)

## predict test inputs and generate ROC data and Calculate AUC (area under curve

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

Y_pred = model.predict(X_valid)
for i in range(len(Y_pred)):
    print(" Predicted=%s" % (Y_pred[i]))

lw = 2 # Largeur de ligne 
# courbes ROC et AUC

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_valid.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= 3

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()




