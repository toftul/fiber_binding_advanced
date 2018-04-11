#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:53:15 2018

@author: ivan
"""

import numpy as np
import lib.const as const
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import gridspec
from basic_units import radians
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import interp2d
#from scipy.interpolate import RectBivariateSpline
#plt.style.use('ggplot')

import matplotlib.mlab as mlab

#%%
# SIGNLE MODE READ

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

z_space, phi_space, Fz, Fphi = np.load('npy_data/F_well_data_sm.npy')
z_eq = 3.44 * 2*np.pi
phi_eq = 0.0

k = 1.0
lam = 2 * np.pi / k
lammmm = 530.0  # [nm] 
R_particle = 100.0 / lammmm * 2 * np.pi / k
fiber_radius = 500.0 / lammmm * 2 * np.pi / k


Fz_interpolated = interp2d(phi_space, z_space, Fz, kind='cubic')
Fphi_interpolated = interp2d(phi_space, z_space, Fphi, kind='cubic')


rho = R_particle + fiber_radius

def U_(phi, z):
    num_phi = np.linspace(0, phi, 300)
    num_z = np.linspace(z_eq, z, 300)
    U = - rho * np.trapz(Fphi_interpolated(num_phi, z), num_phi) - np.trapz(Fz_interpolated(phi, num_z).transpose(), num_z)
    return(U)

z_space = np.linspace(np.min(z_space), np.max(z_space), 400)
phi_space = np.linspace(np.min(phi_space), np.max(phi_space), 300)
U_data = np.zeros([len(z_space), len(phi_space)])
for i, z_i in enumerate(z_space):
    for j, phi_j in enumerate(phi_space):
        U_data[i, j] = U_(phi_j, z_i)
        
U_data -= np.min(U_data) 
kT = const.k_B * 300 * 1e9
U_data /= kT

PHI = phi_space
Z = z_space
PHI, Z = np.meshgrid(PHI, Z)

PHIsm, Zsm, U_data_sm = PHI, Z, U_data

# %%
# MULTIMODE READ


z_space, phi_space, Fz, Fphi = np.load('npy_data/F_well_data_mm.npy')
z_eq = 3.44 * 2*np.pi
phi_eq = 0.0

k = 1.0
lam = 2 * np.pi / k
lammmm = 530.0  # [nm] 
R_particle = 100.0 / lammmm * 2 * np.pi / k
fiber_radius = 500.0 / lammmm * 2 * np.pi / k


Fz_interpolated = interp2d(phi_space, z_space, Fz, kind='cubic')
Fphi_interpolated = interp2d(phi_space, z_space, Fphi, kind='cubic')


rho = R_particle + fiber_radius

def U_(phi, z):
    num_phi = np.linspace(0, phi, 300)
    num_z = np.linspace(z_eq, z, 300)
    U = - rho * np.trapz(Fphi_interpolated(num_phi, z), num_phi) - np.trapz(Fz_interpolated(phi, num_z).transpose(), num_z)
    return(U)

z_space = np.linspace(np.min(z_space), np.max(z_space), 400)
phi_space = np.linspace(np.min(phi_space), np.max(phi_space), 300)
U_data = np.zeros([len(z_space), len(phi_space)])
for i, z_i in enumerate(z_space):
    for j, phi_j in enumerate(phi_space):
        U_data[i, j] = U_(phi_j, z_i)
        
U_data -= np.min(U_data) 
kT = const.k_B * 300 * 1e9
U_data /= kT

PHI = phi_space
Z = z_space
PHI, Z = np.meshgrid(PHI, Z)


PHImm, Zmm, U_data_mm = PHI, Z, U_data


U_data_mm -= np.max(U_data_mm)
U_data_sm -= np.max(U_data_sm)


# %%
# PLOTTING DATA

matplotlib.rcParams.update({'font.size': 14})

levelNum = 90
contourNum = 10
colormap = 'viridis' 
#colormap = 'jet' 

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9))
#fig.tight_layout()
# ## 1st plot
ax = axes.flat[0]

im = ax.contourf(Zsm/(2*np.pi), [val*radians for val in (PHIsm - np.pi)], U_data_sm, levelNum,
             yunits=radians, cmap=colormap, alpha=1.0)
im.set_clim(np.min([U_data_mm, U_data_sm]), np.max([U_data_mm, U_data_sm]))
#CB = plt.colorbar() 
#CB.ax.set_ylabel(r'Trapping potential, $U/kT$')

ax.set_ylabel('Twisting angle, $\phi$, rad')
CS = ax.contour(Zsm/(2*np.pi),[val*radians for val in (PHIsm - np.pi)], U_data_sm, contourNum,
                 colors='k', linewidths=.4,
                 yunits=radians)
levels = np.linspace(np.min(U_data_sm), np.max(U_data_sm), contourNum)
#ax.clabel(CS, 
#          #levels[1::4],  # label every n-th level
#          inline=1,
#          fmt='%.0f',
#          fontsize=10)

# ## 2nd plot
ax = axes.flat[1]

im = ax.contourf(Zmm/(2*np.pi), [val*radians for val in (PHImm - np.pi)], U_data_mm, levelNum,
             yunits=radians, cmap=colormap, alpha=1.0)
im.set_clim(np.min([U_data_mm, U_data_sm]), np.max([U_data_mm, U_data_sm]))
#CB = plt.colorbar() 
#CB.ax.set_ylabel(r'Trapping potential, $U/kT$')

ax.set_xlabel('Distance to the first particle, $\Delta z / \lambda$')
ax.set_ylabel('Twisting angle, $\phi$, rad')
CS = plt.contour(Zmm/(2*np.pi),[val*radians for val in (PHImm - np.pi)], U_data_mm, contourNum,
                 colors='k', linewidths=.4,
                 yunits=radians)
levels = np.linspace(np.min(U_data_mm), np.max(U_data_mm), contourNum)
#ax.clabel(CS, 
#           #levels[1::4],  # label every n-th level
#           inline=1,
#           fmt='%.0f',
#           fontsize=10)

fig.subplots_adjust(right=0.82)
cbar_ax = fig.add_axes([0.85, 0.11, 0.015, 0.77])
CB = fig.colorbar(im, cax=cbar_ax)
CB.set_label('Potential, $U/kT$')
#CB = plt.colorbar(CS, shrink=0.8, extend='both')
#plt.clabel(CS, inline=1, fontsize=10)
#plt.savefig('results/well_contour_two.png')
plt.show()
