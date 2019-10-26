import numpy as np
import glob
import pandas as pd
import random

nuc = np.loadtxt('nucleotide.list.txt', dtype = str)
hemes = np.loadtxt('heme.list.txt', dtype = str)
steroid = np.loadtxt('steroid.list.txt', dtype = str)
control = np.loadtxt('control.list.txt', dtype = str)

voxel_folder = 'deepdrug3d_voxel_data'
files = glob.glob(voxel_folder+'/*.npy')

random.shuffle(files)

targets = []
features = []
steroides = []

for filename in files[:1000]:
    protein_name = filename[22:-4]
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
    elif protein_name in steroid:
        ster = np.load(filename)
        steroides.append(np.reshape(ster,(14, 32, 32, 32)))
    else:
        print (protein_name)
        print ('I can t find this proteine in the lists')
        break
features = np.array(features)
targets = np.array(targets)
steroides = np.array(steroides)

print('features: ', features.shape)
print('targets: ', targets.shape)
print('stero: ', steroides.shape)

np.save('features', features)
np.save('targets', targets)
np.save('steroides', steroides)

