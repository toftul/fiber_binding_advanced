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
# brentq uses the classic Brent’s method to find a zero 
# of the function f on the sign changing interval [a , b]. 
# Generally considered the best of the rootfinding routines.
from scipy.optimize import brentq as root
import scipy.special as sp
# import matplotlib.pyplot as plt


def ha_(beta, k, rf, epsf, epsm):   
    return(rf * np.sqrt(epsf * k*k - beta*beta))

def qa_(beta, k, rf, epsf, epsm):
    return(rf * np.sqrt(beta*beta - epsm * k*k))

# Eigenvalue equations
# for l > 0
def evEqHE(beta, l, k, rf, epsf, epsm):
    ha = ha_(beta, k, rf, epsf, epsm)
    ha2 = ha * ha
    qa = qa_(beta, k, rf, epsf, epsm)
    qa2 = qa * qa
    Jl_1ha = sp.jn(l - 1, ha)
    Jlha = sp.jn(l, ha)
    Klpqa = sp.kvp(l, qa)
    Klqa = sp.kn(l, qa)
    Klpqa_qaKlqa = Klpqa / (qa * Klqa)
    R = np.sqrt(((epsf - epsm)/(2*epsf) * Klpqa_qaKlqa)**2 +
                (l * beta / (np.sqrt(epsf) * k) * (1/qa2 + 1/ha2))**2)
    return(-Jl_1ha / (ha * Jlha) + l/ha2 -
           (epsf + epsm)/(2*epsf) * Klpqa_qaKlqa - R)
         
#    Jl_1ha = sp.jn(l - 1, ha)
#    Jlha = sp.jn(l, ha)
#    Klqa = sp.kn(l, qa)
#    Klpqa = sp.kvp(l, qa)
#    Klpqa_qaKlqa = Klpqa / (qa * Klqa)
#    R = np.sqrt(((epsf - epsm)/(2*epsf) * Klpqa_qaKlqa)**2 +
#                (l * beta / (np.sqrt(epsf) * k) * (1/(qa*qa) + 1/(ha*ha)))**2)
#    return(-Jl_1ha / (ha * Jlha) -
#           (epsf + epsm)/(2*epsf) * Klpqa_qaKlqa + l/(ha*ha) - R)

# for l > 0
def evEqEH(beta, l, k, rf, epsf, epsm):
    ha = ha_(beta, k, rf, epsf, epsm)
    ha2 = ha * ha
    qa = qa_(beta, k, rf, epsf, epsm)
    qa2 = qa * qa
    Jl_1ha = sp.jn(l - 1, ha)
    Jlha = sp.jn(l, ha)
    Klpqa = sp.kvp(l, qa)
    Klqa = sp.kn(l, qa)
    Klpqa_qaKlqa = Klpqa / (qa * Klqa)
    R = np.sqrt(((epsf - epsm)/(2*epsf) * Klpqa_qaKlqa)**2 +
                (l * beta / (np.sqrt(epsf) * k) * (1/qa2 + 1/ha2))**2)
    return(-Jl_1ha / (ha * Jlha) + l/ha2 -
           (epsf + epsm)/(2*epsf) * Klpqa_qaKlqa + R)
    
#    Jl_1ha = sp.jn(l - 1, ha)
#    Jlha = sp.jn(l, ha)
#    Klqa = sp.kn(l, qa)
#    Klpqa = sp.kvp(l, qa)
#    Klpqa_qaKlqa = Klpqa / (qa * Klqa)
#    R = np.sqrt(((epsf - epsm)/(2*epsf) * Klpqa_qaKlqa)**2 +
#                (l * beta / (np.sqrt(epsf) * k) * (1/(qa*qa) + 1/(ha*ha)))**2)
#    return(-Jl_1ha / (ha * Jlha) -
#           (epsf + epsm)/(2*epsf) * Klpqa_qaKlqa + l/(ha*ha) + R)

# for l = 0
def evEqTE(beta, l, k, rf, epsf, epsm):
    ha = ha_(beta, k, rf, epsf, epsm)
    qa = qa_(beta, k, rf, epsf, epsm)
    J1ha = sp.j1(ha)
    J0ha = sp.j0(ha)
    K1qa = sp.k1(qa)
    K0qa = sp.k0(qa)
    return(J1ha / (ha * J0ha) + K1qa / (qa * K0qa))
    
# for l = 0
def evEqTM(beta, l, k, rf, epsf, epsm):
    ha = ha_(beta, k, rf, epsf, epsm)
    qa = qa_(beta, k, rf, epsf, epsm)
    J1ha = sp.j1(ha)
    J0ha = sp.j0(ha)
    K1qa = sp.k1(qa)
    K0qa = sp.k0(qa)
    return(J1ha / (ha * J0ha) + epsm/epsf * K1qa / (qa * K0qa))
    

def eigenValue(foo, l, k, rf, epsf, epsm):
   kMin = k*(np.sqrt(epsm) + 1e-12)
   kMax = k*(np.sqrt(epsf) - 1e-12)
   betaSpace = np.linspace(kMin, kMax, 1500)
   fSpace = foo(betaSpace, l, k, rf, epsf, epsm)
   fSpace = fSpace * (np.abs(fSpace) < .25)
   rPoints = []
   lPoints = []
   for i, f in enumerate(fSpace):
      if i < len(fSpace) - 1:
         if (f == 0) & (fSpace[i + 1] != 0):
            lPoints = np.hstack((lPoints, betaSpace[i + 1]))
         if (f != 0) & (fSpace[i + 1] == 0):
            rPoints = np.hstack((rPoints, betaSpace[i + 1]))
   if (len(lPoints) > 0) & (len(rPoints) > 0):
      if lPoints[0] > rPoints[0]:
         lPoints = np.hstack((kMin, lPoints))
      if lPoints[-1] > rPoints[-1]:
         rPoints = np.hstack((rPoints, kMax))  
   if len(lPoints) == 0:
      lPoints = np.array([kMin])
   if len(rPoints) == 0:
      rPoints = np.array([kMax])
   if (len(rPoints) == 0) & (len(lPoints) == 0):
      lPoints, rPoints = np.array([kMin]), np.array([kMax])
   
   ab = np.zeros([len(lPoints), 2])
   ab[:, 0], ab[:, 1] = lPoints, rPoints

   # f(a) and f(b) must have different signs
   # so delete all intervals [a, b] with the same signs
   numToDel = []
   for i in range(len(ab)):
      if foo(ab[i, 0], l, k, rf, epsf, epsm) * foo(ab[i, 1], l, k, rf, epsf, epsm) > 0:
         numToDel = np.hstack((numToDel, i))
   if len(numToDel) > 0:
      for j, num in enumerate(numToDel):
         if len(ab) == 1:
#            plt.plot(betaSpace, foo(betaSpace, l, k, rf, epsf, epsm))
#            plt.grid()
#            plt.ylim(-2, 2)
#            plt.xlim(kMin, kMax)
#            plt.show()
            return(np.array([0]))
         ab = np.delete(ab, num - j, 0)  
   
   if len(ab) == 0:
#      plt.plot(betaSpace, foo(betaSpace, l, k, rf, epsf, epsm))
#      plt.grid()
#      plt.ylim(-2, 2)
#      plt.xlim(kMin, kMax)
#      plt.show()
      return(np.array([0]))
   else:
      roots = np.zeros(len(ab))
      for i in range(len(ab)):
         r = root(foo, ab[i, 0], ab[i, 1], args=(l, k, rf, epsf, epsm))
         if (r > kMin) & (r < kMax):
            roots[i] = r
         else:
            roots[i] = 0
      roots = roots[~np.isnan(roots)]
      roots[::-1].sort()
#      plt.plot(betaSpace, foo(betaSpace, l, k, rf, epsf, epsm))
#      for interval in ab:
#         plt.scatter(interval, foo(interval, l, k, rf, epsf, epsm))
#      plt.scatter(roots, roots*0, c='red')
#      plt.grid()
#      plt.ylim(-2, 2)
#      plt.xlim(kMin, kMax)
#      plt.show()
      return roots
