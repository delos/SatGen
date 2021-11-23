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


@vectorize([float64(float64,float64,float64)],nopython=True
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

