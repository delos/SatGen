import os
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
  
power_tab_file = os.path.join(os.path.dirname(__file__),'etc/gvdb_power.npy')
log_power_tab = np.log(np.load(power_tab_file))


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

def rhokfun(k,iz,i0,mass,concentration,VirialRadius,phi_res=None):
    # mass, concentration, VirialRadius have shape (N_halo, N_z)

    # accretion redshift-index, shape (N_halo,)
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
    rhok = np.zeros((m0.size,k.size))
    rhok[~ib] = m0[~ib][:,None]*interp(fb[~ib][:,None],ch[~ib][:,None],rh[~ib][:,None]*k[None,:])

    return rhok # still need to square, sum over halos, and divide by spatial volume

def rhokfun0(k,iz,i0,mass,concentration,VirialRadius):
    # mass, concentration, VirialRadius have shape (N_halo, N_z)

    # accretion redshift-index, shape (N_halo,)
    i0_vec = (np.arange(len(i0)),i0)

    # bad halos, shape (N_halo,)
    ib = np.isnan(mass[i0_vec]) | (mass[i0_vec]<=0.)
    ib = ib | np.isnan(concentration[i0_vec]) | np.isnan(VirialRadius[i0_vec])
    ib = ib | (concentration[i0_vec] <= 0.) | (VirialRadius[i0_vec] <= 0.)

    # parameters at accretion, shape (N_halo,)
    m0 = mass[i0_vec]
    ch = concentration[i0_vec]
    rh = VirialRadius[i0_vec]
    
    # shape (N_halo, N_k)
    rhok = np.zeros((m0.size,k.size))
    rhok[~ib] = m0[~ib][:,None]*interp(1.,ch[~ib][:,None],rh[~ib][:,None]*k[None,:])

    return rhok # still need to square, sum over halos, and divide by spatial volume

if __name__ == '__main__':
    from sys import argv
    from glob import glob
    import os

    try:
        directory = argv[1]
    except:
        raise Exception('python script.py <directory> [iz=0] [kmin=0.1,kmax=1158.5,nk=55] [phi_res]')
    
    try: iz = int(argv[2])
    except: iz = 0

    try:
      tmp = argv[3].split(',')
      kmin = float(tmp[0])
      kmax = float(tmp[1])
      nk = int(tmp[2])
    except:
      kmin, kmax, nk = 0.1, 1158.5, 55
    k = np.geomspace(kmin,kmax,nk)

    try: phi_res = float(argv[4])
    except: phi_res = None

    files = glob(directory + '/tree*.npz')

    for file in files:
        print(file,flush=True)
        data = np.load(file)

        # accretion redshift
        i0 = np.argmax(data['mass'],axis=1)
        i0_vec = (np.arange(len(i0)),i0)

        # rho(k)
        rhok = rhokfun(k,iz,i0,data['mass'],data['concentration'],data['VirialRadius'],phi_res=phi_res)
        rhok0 = rhokfun0(k,iz,i0,data['mass'],data['concentration'],data['VirialRadius'])

        # get parent ID and convert to new indices
        ParentID = data['ParentID'][:,iz]
        ParentID0 = data['ParentID'][i0_vec]

        # write
        np.savez(directory + '/rhok' + os.path.basename(file)[4:-4] + '_%d.npz'%iz,
                 k=k, rhok=rhok,rhok0=rhok0,
                 pos=data['coordinates'][:,iz,:3],
                 vel=data['coordinates'][:,iz,3:],
                 redshift=data['redshift'][iz],
                 order=data['order'][:,iz], ParentID=data['ParentID'][:,iz], mass=data['mass'][:,iz],
                 order0=data['order'][i0_vec], ParentID0=data['ParentID'][i0_vec], mass0=data['mass'][i0_vec], 
                 )

