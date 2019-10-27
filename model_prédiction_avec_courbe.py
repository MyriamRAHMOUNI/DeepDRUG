# Lecture des données
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense, Convolution3D, Flatten, MaxPooling3D, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

def creation_model():
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
    # Fully connected layer pour 3 classes
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('sigmoid'))
    return model

def train_model(model, X_train, y_train, X_valid, y_valid):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    EarlyStopping(monitor='val_loss', min_delta=0.1, patience=6, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1)

def prediction(model, x, out, save_results): # x c'est les proteines à prédir out c'est le nom du fichier qui contiendra mes prédictions
    pred = model.predict(x)
    if save_results == 'True':
        name_file = out+'_pred.txt'
        f= open(name_file,"w+")
        for i in range(len(pred)):
            f.write(" Predicted=%s" % (pred[i]))
        f.close() 
    return pred

def Camembert_pred(stéroid_pred): 
    Nuc = 0
    Hem = 0
    con = 0
    indef = 0
    for i in range(len(stéroid_pred)):
        proba = np.array(stéroid_pred[i])
        print(proba)
        Class = (proba > 0.5).astype(np.int)
        print(Class)
        if Class[0] == 1 and Class[1] == 0 and Class[0] == 0:
            Nuc = Nuc + 1
        elif Class[1] == 1 and Class[0] == 0 and Class[2] == 0:
            Hem = Hem + 1
        elif Class[2] == 1 and Class[0] == 0 and Class[1] == 0:
            con = con + 1
        else:
            indef = indef + 1
    labels = 'Nucléotides', 'Hemes', 'Contrôle', 'indéfini'
    sizes = [Nuc, Hem, con, indef]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.legend()
    plt.axis('equal')
    plt.savefig('stéroide_prédiction.png')
    plt.show()
    pass


def ROC_AUC(y_valid, Y_pred):
    lw = 2 # Largeur de ligne 
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

if __name__ == "__main__":
    #Importer mes données
    features = np.load('features.npy')
    targets = np.load('targets.npy')
    steroides = np.load('steroides.npy')
    #Création du model
    model = creation_model()
    ## Training
    X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.4, random_state=42) 
    train_model(model, X_train, y_train, X_valid, y_valid)
    ## predict inputs X_valid et steroides
    Y_pred = prediction(model, X_valid, 'None', 'False') 
    stéroid_pred = prediction(model, steroides, 'steroides', 'True') 
    #afficher mes prédictions stériodes en Camembert
    Camembert_pred(stéroid_pred)
    # courbes ROC et calcule AUC des 3 classes
    ROC_AUC(y_valid, Y_pred)

