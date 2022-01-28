import numpy as np
import os
from glob import glob
from sys import argv

file = argv[1]
data = np.load(file)
file_trunc = file[:-4]
try: os.mkdir(file_trunc)
except: pass

# global data
mass = data['mass']
order = data['order']
VirialRadius = data['VirialRadius']
izroot = mass.argmax(axis=1) # root-redshift ids of all the branches
idx = np.arange(mass.shape[0]) # branch ids of all the branches
levels = np.unique(order[order>=0]) # all >0 levels in the tree
izmax = mass.shape[1] - 1 # highest redshift index
min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0,:] > 0)[-1][0]]
M0 = mass[0,0]
np.savez(file_trunc + '/global.npz',izroot=izroot,idx=idx,levels=levels,izmax=izmax,min_rvir=min_rvir,M0=M0,)

# per-redshift data
fields = list(data)
nz = data['redshift'].size
for iz in range(nz):
    out = {}
    for f in fields:
        ax = np.where(np.array(data[f].shape) == nz)[0][0]
        print(iz,f)
        out[f] = np.take(data[f],iz,ax)
    np.savez(file_trunc + '/%d.npz'%iz,**out)



