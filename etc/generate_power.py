import numpy as np
from scipy.special import sici
from mpmath import *

gvdb_fp = [ 3.37821658e-01, -2.21730464e-04,  1.56793984e-01,
                     1.33726984e+00,  4.47757739e-01,  2.71551083e-01,
                    -1.98632609e-01,  1.05905814e-02, -1.11879075e+00,
                     9.26587706e-02,  4.43963825e-01, -3.46205146e-02,
                    -3.37271922e-01, -9.91000445e-02,  4.14500861e-01]

def dpkdk(x,y,ch,rte,delta):
  return 1 / (1.+x)**2 / (1. + (x * ((ch - rte)/(ch*rte)))**delta) * sin(x*y) / y

def pk_NFW(l):
  if l > 30.:
    return 1. / l**2
  else:
    si,ci = sici(l)
    return (-(np.cos(l)*ci) + (np.sin(l)*(np.pi - 2*si))/2.)

def compute(fb,ch):
  
  f = ch**3 / (np.log(1+ch) - ch/(1.+ch))
  
  rs = 1. / ch
  log10ch = log10(ch)
  log10fb = log10(fb)
  
  fte = float(10**(gvdb_fp[0] * (ch / 10.)**gvdb_fp[1] * log10fb + gvdb_fp[2] * (1. - fb)**gvdb_fp[3] * log10ch))
  rte = 10**(log10ch + gvdb_fp[4] * (ch / 10.)**gvdb_fp[5] * log10fb + gvdb_fp[6] * (1. - fb)**gvdb_fp[7] * log10ch) * exp(gvdb_fp[8] * (ch / 10.)**gvdb_fp[9] * (1. - fb))
  delta = 10**(gvdb_fp[10] + gvdb_fp[11]*(ch / 10.)**gvdb_fp[12] * log10fb + gvdb_fp[13] * (1. - fb)**gvdb_fp[14]* log10ch)
  rte = min(rte, ch)
    
  l_vals = k_vals_int * rs
  
  pk = np.zeros(nk)
  
  for i,l in enumerate(l_vals):
    if l > 30./rte:
      pk[i] = pk_NFW(l)
    elif l > 1./rte:
      pk[i] = float(quadosc(lambda x: dpkdk(x,l,ch,rte,delta),[0,inf],omega=l))
    else:
      pk[i] = float(quad(lambda x: dpkdk(x,l,ch,rte,delta),[0,inf]))
  
  return pk * fte * f / ch**3

nfb = 160
nk = 234
ncs = 40

fb_vals_int = np.logspace(-8, 0, nfb)
k_vals_int = np.logspace(-2.,7.5, nk)
cs_vals_int = np.geomspace(1, 140, ncs)

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

i = rank

fb = fb_vals_int[i]
power = np.zeros((ncs,nk))

for j, cs in enumerate(cs_vals_int):
	power[j,:] = compute(fb,cs)

np.save('_gvdb_power_%d.npy'%i,power)

