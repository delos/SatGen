import numpy as np
from numba import vectorize, float64

eps = 1e-3

nfb = 160
nk = 234
ncs = 40

fb_vals_int = np.logspace(-8, 0, nfb)
k_vals_int = np.logspace(-2.,7.5, nk)
cs_vals_int = np.geomspace(1, 140, ncs)

log_fb_vals_int = np.log(fb_vals_int)
log_k_vals_int  = np.log(k_vals_int)
log_cs_vals_int = np.log(cs_vals_int)
  
log_power_tab = np.log(np.load('etc/gvdb_power.npy'))


@vectorize([float64(float64,float64,float64)],nopython=True)
def interp(fb,cs,kk):
    logfb = np.log(fb)
    logcs = np.log(cs)
    logk = np.log(kk)
    if logfb < log_fb_vals_int[0]:
        logfb = log_fb_vals_int[0]
    elif logfb > log_fb_vals_int[-1] - eps:
        logfb = log_fb_vals_int[-1] - eps
    if logcs < log_cs_vals_int[0]:
        logcs = log_cs_vals_int[0]
    elif logcs > log_cs_vals_int[-1] - eps:
        logcs = log_cs_vals_int[-1] - eps
    if logk < log_k_vals_int[0]:
        logk = log_k_vals_int[0]
    elif logk > log_k_vals_int[-1] - eps:
        logk = log_k_vals_int[-1] - eps

    f = (logfb-log_fb_vals_int[0])/(log_fb_vals_int[-1]-log_fb_vals_int[0]) * (nfb-1)
    i = int(f)
    f -= i

    g = (logcs-log_cs_vals_int[0])/(log_cs_vals_int[-1]-log_cs_vals_int[0]) * (ncs-1)
    j = int(g)
    g -= j

    h = (logk-log_k_vals_int[0])/(log_k_vals_int[-1]-log_k_vals_int[0]) * (nk-1)
    k = int(h)
    h -= k

    ret = 0.

    ret += (1-f)*(1-g)*(1-h) * log_power_tab[ i , j , k ]

    ret +=   f  *(1-g)*(1-h) * log_power_tab[i+1, j , k ]
    ret += (1-f)*  g  *(1-h) * log_power_tab[ i ,j+1, k ]
    ret += (1-f)*(1-g)*  h   * log_power_tab[ i , j ,k+1]

    ret += (1-f)*  g  *  h   * log_power_tab[ i ,j+1,k+1]
    ret +=   f  *(1-g)*  h   * log_power_tab[i+1, j ,k+1]
    ret +=   f  *  g  *(1-h) * log_power_tab[i+1,j+1, k ]

    ret +=   f  *  g  *  h   * log_power_tab[i+1,j+1,k+1]

    return np.exp(ret)

def power(k,iz,mass,concentration,VirialRadius,phi_res=None):
    # mass, concentration, VirialRadius have shape (N_halo, N_z)

    # accretion redshift-index, shape (N_halo,)
    i0 = np.argmax(mass,axis=1)
    i0_vec = (np.arange(len(i0)),i0)

    # bad halos, shape (N_halo,)
    ib = np.isnan(mass[:,iz]) | np.isnan(mass[i0_vec]) | (mass[i0_vec]<=0.) | (mass[:,iz]<=0.) | (iz > i0)
    ib = ib | np.isnan(concentration[i0_vec]) | np.isnan(VirialRadius[i0_vec])
    ib = ib | (concentration[i0_vec] <= 0.) | (VirialRadius[i0_vec] <= 0.)

    # parameters at accretion, shape (N_halo,)
    m0 = mass[i0_vec]
    ch = concentration[i0_vec]
    rh = VirialRadius[i0_vec]
    
    # parameters at redshift[iz], shape (N_halo,)
    fb = mass[:,iz] / m0

    # disrupted halos
    if phi_res is None:
      phi_res = np.min(fb[~ib])
    ib = ib | np.isclose(phi_res,fb)

    # shape (N_halo, N_k)
    pk = np.zeros((m0.size,k.size))
    pk[~ib] = m0[~ib][:,None]*interp(fb[~ib][:,None],ch[~ib][:,None],rh[~ib][:,None]*k[None,:])

    return pk**2 # still need to sum over halos and divide by spatial volume

if __name__ == '__main__':
    from sys import argv
    from glob import glob
    import os

    try:
        directory = argv[1]
    except:
        print('python script.py <directory> [iz=0] [kmin=0.1,kmax=100,nk=100] [phi_res]')
        raise
    
    try: iz = int(argv[2])
    except: iz = 0

    try: kmin, kmax, nk = [float(x) for x in argv[3].split(',')]
    except: kmin, kmax, nk = 0.1, 100., 100
    k = np.geomspace(kmin,kmax,nk)

    try: phi_res = float(argv[4])
    except: phi_res = None

    files = glob(directory + '/tree*.npz')

    for file in files:
        print(file,flush=True)
        data = np.load(file)
        p = power(k,iz,data['mass'],data['concentration'],data['VirialRadius'],phi_res=phi_res)

        # don't write neglected halos
        ig = p.any(axis=1)
        
        # map indices
        nh = p.shape[0]
        invmap = np.arange(nh)[ig] # invmap[new_index] = old_index
        fwdmap = np.zeros(nh,dtype=int)-1
        fwdmap[invmap] = np.arange(len(invmap)) # fwdmap[old_index] = new_index

        # get parent ID and convert to new indices
        ParentID = fwdmap[data['ParentID'][:,iz]][ig]

        # only consider order 1 subhalos
        #ig = ig & (data['order'][:,iz] == 1)

        # write
        np.savez(directory + '/power' + os.path.basename(file)[4:],
                 k=k, power=p[ig], R=data['coordinates'][ig,iz,0],
                 z=data['coordinates'][ig,iz,2], redshift=data['redshift'][iz],
                 order=data['order'][ig,iz], ParentID=ParentID, mass=data['mass'][ig,iz], )

