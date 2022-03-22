import numpy as np
from numba import vectorize, float64

if __name__ == '__main__':
    from sys import argv
    from glob import glob
    import os

    try:
        directory = argv[1]
    except:
        raise Exception('python script.py <directory> [iz=0] [phi_res]')
    
    try: iz = int(argv[2])
    except: iz = 0

    try: phi_res = float(argv[3])
    except: phi_res = None

    files = glob(directory + '/tree*.npz')

    for file in files:
        print(file,flush=True)
        data = np.load(file)
        mass = data['mass']

        # don't write bad halos
        ig = np.isfinite(mass)&(mass>0)
        
        # map indices
        nh = mass.shape[0]
        invmap = np.arange(nh)[ig] # invmap[new_index] = old_index
        fwdmap = np.zeros(nh,dtype=int)-1
        fwdmap[invmap] = np.arange(len(invmap)) # fwdmap[old_index] = new_index

        # get parent ID and convert to new indices
        ParentID = fwdmap[data['ParentID'][:,iz]][ig]

        # only consider order 1 subhalos
        #ig = ig & (data['order'][:,iz] == 1)

        # write
        np.savez(directory + '/mass' + os.path.basename(file)[4:-4] + '_%d.npz'%iz,
                 mass=mass[ig,iz], R=data['coordinates'][ig,iz,0],
                 z=data['coordinates'][ig,iz,2], redshift=data['redshift'][iz],
                 order=data['order'][ig,iz], ParentID=ParentID, )

