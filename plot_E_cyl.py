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
Rff = 500
R_particle = 120

eps_in = 3.5  # 2.09
eps_out = 1.77

k = 1.0
R = Rff/lamm * 2*np.pi
m = np.sqrt(eps_in / eps_out)
E0 = 1.0
nmin = -15
nmax = 15
case = 2

def E(x, y):
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

box_mm = 30.0
box_sm = box_mm #* Rff_sm / Rff_mm

# data for sm
xvec_sm = np.linspace(-box_sm/2, box_sm/2, NDOTS)                               
x_sm, y_sm = np.meshgrid(xvec_sm, xvec_sm)
EEE = E(x_sm, y_sm, R_sm)
z0 = EEE[0]                                
z1 = EEE[1]
z2 = EEE[2]   
zmod = np.sqrt(EEE[0] * EEE[0] + EEE[1] * EEE[1] + EEE[2] * EEE[2])
                             

#plt.contourf(x, y, z0, NDOTS)                             
#plt.colorbar() 
#plt.show()
#
#plt.contourf(x, y, z1, NDOTS)                             
#plt.colorbar() 
#plt.show()
#
#plt.contourf(x, y, z2, NDOTS)                             
#plt.colorbar() 
#plt.show()



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

NDOTS = 40

def get_circle(x0, y0, R):
    phi = np.linspace(-np.pi, np.pi, 200)
    x = x0 + R * np.cos(phi)
    y = y0 + R * np.sin(phi)
    return x, y

#fig = plt.figure(figsize=(10, 5)) 
#fig.tight_layout()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4.7))
#fig.tight_layout()

ax = axes.flat[0]

im = ax.contourf(y_sm/(2*np.pi), -x_sm/(2*np.pi), zmod_sm, NDOTS, cmap='jet')    
#ax.set_title(r'$kR =$ %.1f, $\sqrt{\epsilon_f / \epsilon_m} =$ %.1f' % (k*R_sm, m), fontsize=14)   
ax.set_xlabel('$x/\lambda$', fontsize=14)
ax.set_ylabel('$y/\lambda$', fontsize=14)             

#bar = plt.colorbar(orientation='vertical')
#bar.set_label('Amplitude of the total field, $|E_{sc}|/E_0$', fontsize=14)
im.set_clim(0, np.max(zmod_mm).real)

xCirc, yCirc = get_circle(0, (Rff_sm + R_particle)/lamm, R_particle/lamm)
ax.plot(xCirc, yCirc, 'k--')


ax = axes.flat[1]

im = ax.contourf(y_mm/(2*np.pi), -x_mm/(2*np.pi), zmod_mm, NDOTS, cmap='jet')    
#ax.set_title('$kR =$ %.1f, $\sqrt{\epsilon_f / \epsilon_m} =$ %.1f' % (k*R_mm, m), fontsize=14)   
ax.set_xlabel('$x/\lambda$', fontsize=14)
ax.set_ylabel('$y/\lambda$', fontsize=14)             

#bar = plt.colorbar(orientation='horizontal')
#bar.set_label('Amplitude of the total field, $|E_{sc}|/E_0$', fontsize=14)

xCirc, yCirc = get_circle(0, (Rff_mm + R_particle)/lamm, R_particle/lamm)
ax.plot(xCirc, yCirc, 'k--')

fig.subplots_adjust(bottom=0.27)

cbar_ax = fig.add_axes([0.2, 0.1, 0.59, 0.03])
CB = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
CB.set_ticks(np.array([0, 1, 4, 7]))
CB.set_ticklabels(['0', '1', '4', '7'])
CB.set_label('Amplitude of the total electric field, $|\mathbf{E}_{sc} + \mathbf{E}_0|/E_0$', fontsize=14)

plt.savefig('results/Esc.pdf')
plt.show()