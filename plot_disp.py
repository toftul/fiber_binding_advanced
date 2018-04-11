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
epsf = 3.5
#n1 = np.sqrt(epsf)
epsm = 1.77
#n2 = np.sqrt(epsm)

# signle mode ratio
RfOverLambdaSM = 0.25
# multimode ratio
RfOverLambdaMM = 0.85
# Remark: it is Rf/lam not Rf/lam0 !
# convert into Rf/lam0
RfOverLambdaSM, RfOverLambdaMM = RfOverLambdaSM/np.sqrt(epsm), RfOverLambdaMM/np.sqrt(epsm)

omegaRf = 0.8
rf = 2*np.pi
omegaSpace = np.linspace(0, omegaRf, 300)

# %%

lMax = 5
mMax = 15
# calc HE family
betaSpaceHEFamily = np.zeros([lMax, mMax, len(omegaSpace)])
for l in range(lMax):
   for i, omega in enumerate(omegaSpace):
      roots = fd.eigenValue(fd.evEqHE, l, omega, rf, epsf, epsm)
      for m, rt in enumerate(roots):
         betaSpaceHEFamily[l, m, i] = rt

# calc EH fimily 
betaSpaceEHFamily = np.zeros([lMax, mMax, len(omegaSpace)])
for l in range(lMax):
   for i, omega in enumerate(omegaSpace):
      roots = fd.eigenValue(fd.evEqEH, l, omega, rf, epsf, epsm)
      for m, rt in enumerate(roots):
         betaSpaceEHFamily[l, m, i] = rt
         
# calc TE fimily 
betaSpaceTEFamily = np.zeros([mMax, len(omegaSpace)])
l = 0
for i, omega in enumerate(omegaSpace):
   roots = fd.eigenValue(fd.evEqTE, l, omega, rf, epsf, epsm)
   for m, omega in enumerate(roots):
      betaSpaceTEFamily[m, i] = rt
      
# calc TM fimily 
betaSpaceTMFamily = np.zeros([mMax, len(omegaSpace)])
l = 0
for i, omega in enumerate(omegaSpace):
   roots = fd.eigenValue(fd.evEqTM, l, omega, rf, epsf, epsm)
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

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)


# Plotting w(k)
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(6.5, 4.))

# plot prm line
x = np.linspace(.04, 1.3)
plt.plot(x, RfOverLambdaMM * x/x, 'k', linewidth=.5)
label = 'MM, $R_f/\lambda_0=%.2f$'%(RfOverLambdaMM)
plt.text(0.05, RfOverLambdaMM + .04, label,
         horizontalalignment='left')
x = np.linspace(.04, .85)
plt.plot(x, RfOverLambdaSM * x/x, 'k', linewidth=.5)
label = 'SM, $R_f/\lambda_0=%.2f$'%(RfOverLambdaSM)
plt.text(0.35, RfOverLambdaSM - .08, label,
         horizontalalignment='left')

# light and fiber lines
plt.plot(np.sqrt(epsm)*omegaSpace, omegaSpace, 'k--', alpha=.5)
plt.plot(np.sqrt(epsf)*omegaSpace, omegaSpace, 'k--', alpha=.5)
xTextLight = .55
yTextLight = xTextLight / np.sqrt(epsm) + .12
plt.text(xTextLight, yTextLight, r'$\omega = c\beta / n_m$',
         horizontalalignment='center', rotation=np.arctan(1./np.sqrt(epsm)) * 180/np.pi)
xTextLight = 1.1
yTextLight = xTextLight / np.sqrt(epsf) - .02
plt.text(xTextLight, yTextLight, r'$\omega = c\beta / n_f$',
         horizontalalignment='center', rotation=np.arctan(1./np.sqrt(epsf)) * 180/np.pi)

# modes
for j in convenientOrder[0, :]:
    plt.plot(betaSpaceFamily[j], omegaSpace, next(linecycler), label=nameSpace[j])

plt.legend(ncol=2, frameon=False, fontsize=10)
plt.xlabel(r'$\beta/k_f$')
plt.ylabel(r'$\omega/\omega_f =  R_f / \lambda_0 = R_f / n_m \lambda$')
#plt.grid()

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')

#plt.xlim(0, np.max(np.sqrt(epsf)*omegaSpace))
plt.xlim(0, 1.6)
plt.ylim(0, np.max(omegaSpace))


plt.savefig('results/fiberDispersion.pdf')
plt.show()


