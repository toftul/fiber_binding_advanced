#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:30:08 2018

@author: ivan
"""

import numpy as np
from scipy.interpolate import interp1d
import const as const

# Ag
data = np.loadtxt('refractiveindex/AgWerner.txt')

wl = data[:150, 0] * 1e-6
n = data[:150, 1]
k = data[-150:, 1]

epsReAg = n*n - k*k
epsImAg = 2 * n * k

epsReAgFunc = interp1d(wl, epsReAg, kind='cubic')
epsImAgFunc = interp1d(wl, epsImAg, kind='cubic')

def epsilonAg(wl0):
    return(epsReAgFunc(wl0) + 1j * epsImAgFunc(wl0))


# Au
data = np.loadtxt('refractiveindex/AuJohnson.txt')

wl = data[:49, 0] * 1e-6
n = data[:49, 1]
k = data[-49:, 1]

epsReAu = n*n - k*k
epsImAu = 2 * n * k

epsReAuFunc = interp1d(wl, epsReAu, kind='cubic')
epsImAuFunc = interp1d(wl, epsImAu, kind='cubic')

def epsilonAu(wl0):
    return(epsReAuFunc(wl0) + 1j * epsImAuFunc(wl0))


# Al
data = np.loadtxt('refractiveindex/AlRakic.txt')

wl = data[:206, 0] * 1e-6
n = data[:206, 1]
k = data[-206:, 1]

epsReAl = n*n - k*k
epsImAl = 2 * n * k

epsReAlFunc = interp1d(wl, epsReAl, kind='cubic')
epsImAlFunc = interp1d(wl, epsImAl, kind='cubic')

def epsilonAl(wl0):
    return(epsReAlFunc(wl0) + 1j * epsImAlFunc(wl0))


# Si
data = np.loadtxt('refractiveindex/SiGreen.txt')

wl = data[:121, 0] * 1e-6
n = data[:121, 1]
k = data[-121:, 1]

epsReSi = n*n - k*k
epsImSi = 2 * n * k

epsReSiFunc = interp1d(wl, epsReSi, kind='cubic')
epsImSiFunc = interp1d(wl, epsImSi, kind='cubic')

def epsilonSi(wl0):
    return(epsReSiFunc(wl0) + 1j * epsImSiFunc(wl0))
    
    
# Cu
data = np.loadtxt('refractiveindex/CuJohnson.txt')

wl = data[:121, 0] * 1e-6
n = data[:121, 1]
k = data[-121:, 1]

epsReCu = n*n - k*k
epsImCu = 2 * n * k

epsReCuFunc = interp1d(wl, epsReCu, kind='cubic')
epsImCuFunc = interp1d(wl, epsImCu, kind='cubic')

def epsilonCu(wl0):
    return(epsReCuFunc(wl0) + 1j * epsImCuFunc(wl0))
    

# Drude
wp = 1
gamma = 1

def epsilon(wl0, material):
    if material == 'Au':
        return(epsilonAu(wl0))
    elif material == 'Ag':
        return(epsilonAg(wl0))
    elif material == 'Al':
        return(epsilonAl(wl0))
    elif material == 'Si':
        return(epsilonSi(wl0))
    elif material == 'Cu':
        return(epsilonCu(wl0))
    else:
        w = 2*np.pi*const.c/wl0
        return(1 - wp*wp/(w * (w + 1j * gamma)))
    

#import matplotlib.pyplot as plt
#
#wl = np.linspace(300, 800) * 1e-9
#
#plt.plot(wl*1e9, epsilonAg(wl).real, label='Re')
#plt.plot(wl*1e9, epsilonAg(wl).imag, label='Im')
#plt.plot(wl*1e9, epsilonSi(wl).real, label='Re')
#plt.plot(wl*1e9, epsilonSi(wl).imag, label='Im')
#plt.plot(wl*1e9, epsilonAu(wl).real, label='Re')
#plt.plot(wl*1e9, epsilonAu(wl).imag, label='Im')
#plt.legend()
#plt.plot()
