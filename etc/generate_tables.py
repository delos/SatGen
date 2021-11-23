import numpy as np
from numba import njit
from scipy.integrate import quad

nfb = 160
nr = 171
ncs = 40

fb_vals_int = np.logspace(-8, 0, nfb)
r_vals_int = np.logspace(-7.5, 2., nr)
cs_vals_int = np.geomspace(1, 140, ncs)

gvdb_fp = np.array([ 3.37821658e-01, -2.21730464e-04,  1.56793984e-01,
                     1.33726984e+00,  4.47757739e-01,  2.71551083e-01,
                    -1.98632609e-01,  1.05905814e-02, -1.11879075e+00,
                     9.26587706e-02,  4.43963825e-01, -3.46205146e-02,
                    -3.37271922e-01, -9.91000445e-02,  4.14500861e-01])
@njit
def transfer(x,ch,fb,log10ch,log10fb):
  fte = 10**(gvdb_fp[0] * (ch / 10.)**gvdb_fp[1] * log10fb + gvdb_fp[2] * (1. - fb)**gvdb_fp[3] * log10ch)
  rte = 10**(log10ch + gvdb_fp[4] * (ch / 10.)**gvdb_fp[5] * log10fb + gvdb_fp[6] * (1. - fb)**gvdb_fp[7] * log10ch) * np.exp(gvdb_fp[8] * (ch / 10.)**gvdb_fp[9] * (1. - fb))
  delta = 10**(gvdb_fp[10] + gvdb_fp[11]*(ch / 10.)**gvdb_fp[12] * log10fb + gvdb_fp[13] * (1. - fb)**gvdb_fp[14]* log10ch)

  rte = min(rte, ch)

  return fte / (1. + (x * ((ch - rte)/(ch*rte)))**delta)
@njit
def rho(x,rho0,ch,fb,log10ch,log10fb):
  return rho0 / (x * (1.+x)**2.) * transfer(x,ch,fb,log10ch,log10fb)
  
@njit
def dMdx(x,rho0,ch,fb,log10ch,log10fb):
  return rho(x,rho0,ch,fb,log10ch,log10fb) * 4*np.pi*x**2

@njit
def interp(x1,lgx,lgy):
  f = (np.log10(x1)-lgx[0])/(lgx[-1]-lgx[0]) * (len(lgx)-1)
  i = int(f)
  f -= i
  return 10**(lgy[i] * (1-f) + lgy[i+1] * f)

@njit
def sigma_int(x,rho0,ch,fb,log10ch,log10fb,lgx,lgM):
  return rho(x,rho0,ch,fb,log10ch,log10fb) * interp(x,lgx,lgM) / x**2

def compute(fb,cs):
  Mi = 1
  rh = 1
  
  rho0 = Mi / (4*np.pi*rh**3) * cs**3 / (np.log(1+cs) - cs/(1.+cs))
  rs = rh / cs
  log10cs = np.log10(cs)
  log10fb = np.log10(fb)
  
  r_vals = np.concatenate((r_vals_int,r_vals_int[1:32] * r_vals_int[-1]/r_vals_int[0]))
  nr2 = len(r_vals)
  
  x = r_vals / rs
  
  M = np.zeros(nr2)
  
  M[0] = quad(dMdx,0,x[0],args=(rho0,cs,fb,log10cs,log10fb))[0]
  for i in range(1,nr2):
    M[i] = M[i-1] + quad(dMdx,x[i-1],x[i],args=(rho0,cs,fb,log10cs,log10fb),epsabs=0,limit=5000,epsrel=1e-10)[0]
    
  p = 4*np.pi*rho(x,rho0,cs,fb,log10cs,log10fb) - 2*M/x**3
  
  lgx = np.log10(x)
  lgM = np.log10(M)
  
  s = np.zeros(nr2)
  
  s[-1] = 0
  for i in range(nr2-2,-1,-1):
    s[i] = s[i+1] + quad(sigma_int,x[i+1],x[i],args=(rho0,cs,fb,log10cs,log10fb,lgx,lgM),epsabs=0,limit=5000,epsrel=1e-10)[0]
  
  s = np.sqrt(-1./rho(x,rho0,cs,fb,log10cs,log10fb) * s)
  
  return M[:nr]/cs**3,s[:nr]/cs,p[:nr]

gvdb_mm = np.zeros((nfb,ncs,nr))
gvdb_sm = np.zeros((nfb,ncs,nr))
gvdb_pm = np.zeros((nfb,ncs,nr))

for i, fb in enumerate(fb_vals_int):
  print(i)
  for j, cs in enumerate(cs_vals_int):
    gvdb_mm[i,j,:], gvdb_sm[i,j,:], gvdb_pm[i,j,:] = compute(fb,cs)
    
np.save('gvdb_mm.npy',gvdb_mm)
np.save('gvdb_sm.npy',gvdb_sm)
np.save('gvdb_pm.npy',gvdb_pm)

