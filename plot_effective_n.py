#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:42:37 2018

@author: ivan
"""

# Theory here:
# Le Kien, Fam, et al. 
# "Higher-order modes of vacuum-clad ultrathin optical fibers." 
# Physical Review A 96.2 (2017): 023835.

import numpy as np
import matplotlib.pyplot as plt
import lib.fiberDispersion as fd
  

# %%

# permittivities 
epsf = 1.4537 ** 2
#n1 = np.sqrt(epsf)
epsm = 1.0
#n2 = np.sqrt(epsm)

# wavelength wl = wl0 / sqrt(eps_m)
wlnm = 780  # nm
# fiber radius
rfnmSpace = np.linspace(0, 1000, 500)  # nm

# relative radius
ratioSpace = rfnmSpace / wlnm

# nondimensionalization
k = 1.0
wl = 2*np.pi / k
rfSpace = ratioSpace * wl

kMin = k*np.sqrt(epsm)
kMax = k*np.sqrt(epsf)


# %%

#l = 1
#rf = 7.0
#betaSpace = np.linspace(kMin, kMax, 300)
#plt.plot(betaSpace, evEqHE(betaSpace, l, k, rf, epsf, epsm), label='HE')
#plt.plot(betaSpace, evEqEH(betaSpace, l, k, rf, epsf, epsm), label='EH')
#plt.grid()
#plt.ylim(-2, 2)
#plt.legend()
#plt.show()

# %%
# calc HE family
lMax = 5
mMax = 15
betaSpaceHEFamily = np.zeros([lMax, mMax, len(rfSpace)])
for l in range(lMax):
   for i, rf in enumerate(rfSpace):
      roots = fd.eigenValue(fd.evEqHE, l, k, rf, epsf, epsm)
      for m, rt in enumerate(roots):
         betaSpaceHEFamily[l, m, i] = rt

# calc EH fimily 
betaSpaceEHFamily = np.zeros([lMax, mMax, len(rfSpace)])
for l in range(lMax):
   for i, rf in enumerate(rfSpace):
      roots = fd.eigenValue(fd.evEqEH, l, k, rf, epsf, epsm)
      for m, rt in enumerate(roots):
         betaSpaceEHFamily[l, m, i] = rt
         
# calc TE fimily 
betaSpaceTEFamily = np.zeros([mMax, len(rfSpace)])
l = 0
for i, rf in enumerate(rfSpace):
   roots = fd.eigenValue(fd.evEqTE, l, k, rf, epsf, epsm)
   for m, rt in enumerate(roots):
      betaSpaceTEFamily[m, i] = rt
      
# calc TM fimily 
betaSpaceTMFamily = np.zeros([mMax, len(rfSpace)])
l = 0
for i, rf in enumerate(rfSpace):
   roots = fd.eigenValue(fd.evEqTM, l, k, rf, epsf, epsm)
   for m, rt in enumerate(roots):
      betaSpaceTMFamily[m, i] = rt

# write every plot in one array
betaSpaceFamily = []
nameSpace = []
# HE and EH
for l in range(1, lMax):
   for m in range(mMax):
      if np.sum(betaSpaceHEFamily[l, m, :]) > 0.1:
         if len(betaSpaceFamily) == 0:
             betaSpaceFamily = betaSpaceHEFamily[l, m, :]
             nameSpace.append('HE%d%d'%(l,m+1))
         else:
             betaSpaceFamily = np.vstack((betaSpaceFamily, betaSpaceHEFamily[l, m, :]))
             nameSpace.append('HE%d%d'%(l,m+1))
      if (np.sum(betaSpaceEHFamily[l, m, :]) > 0.1):
         if len(betaSpaceFamily) == 0:
             betaSpaceFamily = betaSpaceEHFamily[l, m, :]
             nameSpace.append('EH%d%d'%(l,m+1))
         else:
             betaSpaceFamily = np.vstack((betaSpaceFamily, betaSpaceEHFamily[l, m, :]))
             nameSpace.append('EH%d%d'%(l,m+1))
# TE and TM
l = 0
for m in range(mMax):
   if np.sum(betaSpaceTEFamily[m, :]) > 0.1:
      betaSpaceFamily = np.vstack((betaSpaceFamily, betaSpaceTEFamily[m, :]))
      nameSpace.append('TE%d%d'%(l,m+1))
   if np.sum(betaSpaceTMFamily[m, :]) > 0.1:
      betaSpaceFamily = np.vstack((betaSpaceFamily, betaSpaceTMFamily[m, :]))
      nameSpace.append('TM%d%d'%(l,m+1))
      
convenientOrder = np.zeros([2, len(betaSpaceFamily[:, 0])], dtype=int)
convenientOrder[0, :] = np.arange(0, len(betaSpaceFamily[:, 0]))
for j, family in enumerate(betaSpaceFamily):
    # first place where is nonzero element
    convenientOrder[1, j] = np.nonzero(family)[0][0]

# sort all plot by the first nonzeros element
convenientOrder = convenientOrder[:, convenientOrder[1, :].argsort()]
# array looks like this:
# [[indexOf1stPlot, indexOf2ndPlot, ...],
#  [firstNonzeroIndexOf1stPlot, firstNonzeroIndexOf2ndPlot, ...]]

# zeros to nan
betaSpaceFamily = betaSpaceFamily.astype('float')
betaSpaceFamily[betaSpaceFamily == 0] = np.nan

# %%
# Plotting
#plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(8, 4.5))

for j in convenientOrder[0, :]:
    plt.plot(rfSpace * wlnm/wl, betaSpaceFamily[j], label=nameSpace[j])

plt.legend(ncol=2)
plt.xlabel(r'Fiber radius $R_{f}$, nm')
plt.ylabel(r'Relative propagation constant $\beta/k$')
plt.grid()
plt.ylim(kMin, kMax)
plt.show()


