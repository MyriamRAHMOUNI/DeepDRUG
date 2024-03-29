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
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=25)

## predict test inputs and generate ROC data and Calculate AUC (area under curve

#Y_pred = model.predict(X_valid[:4])
#for i in range(len(4)):
#    print(" Predicted=%s" % (Y_control[i]))


#https://github.com/Tony607/ROC-Keras/blob/master/ROC-Keras.ipynb



