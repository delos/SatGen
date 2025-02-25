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
        raise Exception('python script.py <directory> [max z=0.155] [min mass=1e5]')
    
    try: zmax = int(argv[2])
    except: zmax = 0.155

    try: mmin = float(argv[3])
    except: mmin = None

    print(zmax,mmin)

    files = glob(directory + '/tree*.npz')

    for file in files:
        print(file,flush=True)
        data = np.load(file)

        # mass, concentration, VirialRadius have shape (N_halo, N_z)
        mass = data['mass']
        concentration = data['concentration']
        VirialRadius = data['VirialRadius']
        order = data['order']
        x = data['coordinates'][:,:,:3]
        v = data['coordinates'][:,:,3:]

        # accretion redshift
        i0 = np.nanargmax(data['mass'],axis=1)
        i0_vec = (np.arange(len(i0)),i0)

        # parameters at accretion, shape (N_halo,)
        m0 = mass[i0_vec]
        ch = concentration[i0_vec]
        rh = VirialRadius[i0_vec]

        # redshift range
        zix = data['redshift'] < zmax

        # bad data at accretion, shape (N_halo,)
        ib = ~((m0>mmin)&(ch>0.)&(rh>0.))

        # field halos
        ibf = np.all(x==0.,axis=2)&np.all(v==0.,axis=2)
        mass[ibf] = np.nan
        x[ibf] = np.inf

        # merged/disrupted halos
        ibm = (~(mass>mmin))&(x[:,:,0]>0)
        mass[ibm] = np.nan
        x[ibm] = 0.

        # order=1 only
        ib1 = (order != 1)
        mass[ib1] = np.nan
        x[ib1] = np.nan

        # prune halos
        ib = ib | np.all(ibf[:,zix],axis=1) | np.all(ibm[:,zix],axis=1) | np.all(ib1[:,zix],axis=1)

        # write
        np.savez(directory + '/top' + os.path.basename(file)[4:-4] + '_%.3f.npz'%zmax,
                 M=mass[~ib][:,zix].astype(np.float32),
                 M0=m0[~ib].astype(np.float32),
                 c=ch[~ib].astype(np.float32),
                 R=rh[~ib].astype(np.float32),
                 x=x[~ib][:,zix].astype(np.float32),
                 v=v[~ib][:,zix].astype(np.float32),
                 z=data['redshift'][zix].astype(np.float32),
                 z0=data['redshift'][i0[~ib]].astype(np.float32),
                 )

