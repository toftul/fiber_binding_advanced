#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:20:19 2018

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
import Mie_scat_cyl

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

lamm = 530.0

Rff_sm = 130  # 130 or 495
Rff_mm = 450  # 130 or 495

R_particle = 120

eps_in = 3.5  # 2.09
eps_out = 1.77

k = 1.0
R_sm = Rff_sm/lamm * 2*np.pi
R_mm = Rff_mm/lamm * 2*np.pi
m = np.sqrt(eps_in / eps_out)
E0 = 1.0
nmin = -15
nmax = 15
case = 2

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

NDOTS = 150

box_mm = 30.0
box_sm = box_mm #* Rff_sm / Rff_mm

# data for sm
xvec_sm = np.linspace(-box_sm/2, box_sm/2, NDOTS)                               
x_sm, y_sm = np.meshgrid(xvec_sm, xvec_sm)
EEE = E(x_sm, y_sm, R_sm)
z0 = EEE[0]                                
z1 = EEE[1]
z2 = EEE[2]   

zmod_sm = EEE[0].conjugate() * EEE[0] + \
          EEE[1].conjugate() * EEE[1] + \
          EEE[2].conjugate() * EEE[2]



# data for mm
xvec_mm = np.linspace(-box_mm/2, box_mm/2, NDOTS)                               
x_mm, y_mm = np.meshgrid(xvec_mm, xvec_mm)
EEE = E(x_mm, y_mm, R_mm)
z0 = EEE[0]                                
z1 = EEE[1]
z2 = EEE[2]   

zmod_mm = EEE[0].conjugate() * EEE[0] + \
          EEE[1].conjugate() * EEE[1] + \
          EEE[2].conjugate() * EEE[2]

#zmod = np.sqrt(EEE[0].real * EEE[0].real + 
#               EEE[1].real * EEE[1].real + 
#               EEE[2].real * EEE[2].real)**2
               


# %% 
# plotting part

fig = plt.figure(figsize=(12, 4)) 
fig.tight_layout()

plt.subplot(121)
plt.contourf(x_sm/(2*np.pi), y_sm/(2*np.pi), zmod_sm, NDOTS)    
plt.title('$kR =$ %.1f, $\sqrt{\epsilon_f / \epsilon_m} =$ %.1f' % (k*R_sm, m), fontsize=14)   
plt.xlabel('$x/\lambda$', fontsize=14)
plt.ylabel('$y/\lambda$', fontsize=14)             

bar = plt.colorbar(orientation='vertical')
bar.set_label('Amplitude of the total field, $|E_{sc}|/E_0$', fontsize=14)

plt.plot(- (Rff_sm + R_particle)/lamm, 0, 'ro')


plt.subplot(122)
plt.contourf(x_mm/(2*np.pi), y_mm/(2*np.pi), zmod_mm, NDOTS)    
plt.title('$kR =$ %.1f, $\sqrt{\epsilon_f / \epsilon_m} =$ %.1f' % (k*R_mm, m), fontsize=14)   
plt.xlabel('$x/\lambda$', fontsize=14)
plt.ylabel('$y/\lambda$', fontsize=14)             

bar = plt.colorbar(orientation='vertical')
bar.set_label('Amplitude of the total field, $|E_{sc}|/E_0$', fontsize=14)

plt.plot(- (Rff_mm + R_particle)/lamm, 0, 'ro')


plt.absolute_importsubplots_adjust(left=None, bottom=None, right=None, top=10000,
                wspace=100, hspace=100)
plt.show()