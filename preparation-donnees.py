import numpy as np
import glob
import pandas as pd
import random

def preparation(files, nuc, hemes, steroid, control):
    random.shuffle(files)

    targets = []
    features = []
    steroides = []

    for filename in files[:10]:
        protein_name = filename[22:-4]
        # hemes, control et nuc vont servir de class
        if protein_name in nuc:
            features.append(np.load(filename))
            target = [1, 0, 0]
            targets.append(target)
        elif protein_name in hemes:
            features.append(np.load(filename))
            target = [0, 1, 0]
            targets.append(target)
        elif protein_name in control:
            con = np.load(filename)
            features.append(np.reshape(con,(14, 32, 32, 32)))
            target = [0, 0, 1]
            targets.append(target)
	    # Steroid est crée pour evaluer le model
        elif protein_name in steroid:
            ster = np.load(filename)
            steroides.append(np.reshape(ster,(14, 32, 32, 32)))
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
    np.save('features', features)
    np.save('targets', targets)
    np.save('steroides', steroides)


if __name__ == "__main__":
    nuc = np.loadtxt('nucleotide.list.txt', dtype = str)
    hemes = np.loadtxt('heme.list.txt', dtype = str)
    steroid = np.loadtxt('steroid.list.txt', dtype = str)
    control = np.loadtxt('control.list.txt', dtype = str)

    voxel_folder = 'deepdrug3d_voxel_data'
    files = glob.glob(voxel_folder+'/*.npy')
    preparation(files, nuc, hemes, steroid, control)




