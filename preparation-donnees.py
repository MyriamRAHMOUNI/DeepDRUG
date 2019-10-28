import numpy as np
import glob
import pandas as pd
import random

def preparation(files, nuc, hemes, steroid, control):
    random.shuffle(files)

    targets = []
    features = []
    steroides = []
    i = 0
    j = 0
    k = 0
    l = 0
    for filename in files:
        protein_name = filename[22:-4]
            # hemes, control et nuc vont servir de class        
        if protein_name in nuc:
            if i < 500: 
                features.append(np.load(filename))
                target = [1, 0, 0]
                targets.append(target)
                i = i + 1
        elif protein_name in hemes:
             if j < 500:
                features.append(np.load(filename))
                target = [0, 1, 0]
                targets.append(target)
                j = j + 1
        elif protein_name in control:
            if k < 500:
                con = np.load(filename)
                features.append(np.reshape(con,(14, 32, 32, 32)))
                target = [0, 0, 1]
                targets.append(target)
                k = k + 1
            # Steroid est crée pour evaluer le model
        elif protein_name in steroid:
            if l < 69: 
                ster = np.load(filename)
                steroides.append(np.reshape(ster,(14, 32, 32, 32)))
                l = l + 1
        else:
            raise Exception(protein_name, "Cette protèine n'est présente dans aucune liste" )

    features = np.array(features)
    targets = np.array(targets)
    steroides = np.array(steroides)

#Print shapes (Vérification)
    print('features: ', features.shape)
    print('targets: ', targets.shape)
    print('stero: ', steroides.shape)

#Save results
    np.save('features_test', features)
    np.save('targets_test', targets)
    np.save('steroides_test', steroides)


if __name__ == "__main__":
    nuc = np.loadtxt('nucleotide.list.txt', dtype = str)
    hemes = np.loadtxt('heme.list.txt', dtype = str)
    steroid = np.loadtxt('steroid.list.txt', dtype = str)
    control = np.loadtxt('control.list.txt', dtype = str)

    voxel_folder = 'deepdrug3d_voxel_data'
    files = glob.glob(voxel_folder+'/*.npy')
    preparation(files, nuc, hemes, steroid, control)
