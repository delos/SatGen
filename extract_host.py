import os
import numpy as np
from numba import vectorize, float64

if __name__ == '__main__':
    from sys import argv
    from glob import glob
    import os

    try:
        directory = argv[1]
    except:
        raise Exception('python script.py <directory>')
    
    files = glob(directory + '/tree*.npz')

    for file in files:
        print(file,flush=True)
        data = np.load(file)

        # main host
        ih = np.argmin(np.abs(data['order']),axis=0)
        ih_vec = (ih,np.arange(len(ih)))

        # mass, concentration, VirialRadius have shape (N_halo, N_z)
        mh = data['mass'][ih_vec]
        ch = data['concentration'][ih_vec]
        rh = data['VirialRadius'][ih_vec]

        # write
        np.savez(directory + '/host' + os.path.basename(file)[4:-4],
                 M=mh.astype(np.float32),
                 c=ch.astype(np.float32),
                 R=rh.astype(np.float32),
                 z=data['redshift'].astype(np.float32),
                 )

