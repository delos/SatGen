import numpy as np

nfb = 160
nk = 234
ncs = 40

fb_vals_int = np.logspace(-8, 0, nfb)
k_vals_int = np.logspace(-2.,7.5, nk)
cs_vals_int = np.geomspace(1, 140, ncs)

power = np.zeros((nfb,ncs,nk))

for i,fb in enumerate(fb_vals_int):
  power[i] = np.load('_gvdb_power_%d.npy'%i)

np.save('gvdb_power.npy',power)
