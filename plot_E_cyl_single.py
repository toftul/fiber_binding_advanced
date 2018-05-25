#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:20:19 2018

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
import lib.MieScatCyl as Mie_scat_cyl

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

lamm = 530.0
Rff = 500
R_particle = 120

eps_in = 30.0  # 2.09
eps_out = 1.

k = 1.0
R = Rff/lamm * 2*np.pi
m = np.sqrt(eps_in / eps_out)
E0 = 1.0
nmin = -15
nmax = 15
case = 1

def E(x, y, R):
    if case == 1:
        E0x = 0j
        E0y = 0j
        E0z = E0 * np.exp(- 1j * k * x)
    elif case == 2:
        E0x = 0j
        E0y = E0 * np.exp(- 1j * k * x)
        E0z = 0j
    rho, phi = cart2pol(x, y)
    Ecrho, Ecphi, Ecz = Mie_scat_cyl.Es2(rho, phi, 0.0, k, R, m, E0, nmin, nmax, case)
    Ex = Ecrho * np.cos(phi) - Ecphi * np.sin(phi)
    Ey = Ecrho * np.sin(phi) + Ecphi * np.cos(phi)
    Ez = Ecz
    Ex += E0x
    Ey += E0y
    Ez += E0z
    cut = np.sqrt(x*x + y*y) > R
    cut = cut * np.ones(len(cut))
    return(Ex * cut, Ey * cut, Ez * cut)

NDOTS = 300

box = 50.0

xvec = np.linspace(-box/2, box/2, NDOTS)                               
x, y = np.meshgrid(xvec, xvec)
EEE = E(x, y, R)
z0 = EEE[0]                                
z1 = EEE[1]
z2 = EEE[2]   
zmod = np.sqrt(EEE[0].conjugate() * EEE[0] + 
               EEE[1].conjugate() * EEE[1] + 
               EEE[2].conjugate() * EEE[2])                      


# %%
# Single map
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(6.5, 4.8))
plt.contourf(x / (2*np.pi), y / (2*np.pi), zmod, NDOTS)
plt.xlabel('$x/\lambda$', fontsize=14)
plt.ylabel('$y/\lambda$', fontsize=14)     
CB = plt.colorbar(format='%.1f') 
CB.set_ticks(np.array([np.min(zmod), 1, np.max(zmod)]))    
CB.set_label('$|\mathbf{E}_{sc} + \mathbf{E}_0|/E_0$', fontsize=14)                        
plt.savefig("results/scat.png", bbox_inches='tight', dpi=200)
plt.show()