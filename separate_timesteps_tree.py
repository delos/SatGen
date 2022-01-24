import numpy as np
import os
from glob import glob
from sys import argv

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

filenames = glob(argv[1] + '/tree*.npz')
Ntree = len(filenames)
count = int(np.ceil(Ntree / size))

for i in range(rank*count,(rank+1)*count):
    if i >= Ntree:
        break
    print('[MPI: worker %d on tree %d/%d]'%(rank,i,Ntree),flush=True)

    file = filenames[i]
    data = np.load(file)
    file_trunc = file[:-4]
    try: os.mkdir(file_trunc)
    except: pass

    fields = list(data)

    nz = data['redshift'].size

    for iz in range(nz):
        out = {}
        for f in fields:
            ax = np.where(np.array(data[f].shape) == nz)[0][0]
            print(iz,f)
            out[f] = np.take(data[f],iz,ax)
        np.savez(file_trunc + '/%d.npz'%iz,**out)



