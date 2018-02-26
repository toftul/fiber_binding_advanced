#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:13:42 2018

@author: ivan
"""
import numpy as np
import const
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


k = 1.0
lam = 2 * np.pi / k
lammmm = 400.0  # [nm] 
R_particle = 150.0 / lammmm * 2 * np.pi / k
fiber_radius = 150.0 / lammmm * 2 * np.pi / k

eps_particle = 2.5
eps_out = 1.77
eps_in = 2.09

V = k * fiber_radius * np.sqrt(eps_in - eps_out)

P_laser = 100e-3  # [W]
R_focus = 1e-6  # [m]
Intensity = P_laser / (np.pi * R_focus**2)  # [W/m^2]
E0_mod_real = np.sqrt(0.5 * const.Z0 * Intensity)  # [V/m]


z_space, Fz_TE = np.load('npy_data/Fz_TE_sm.npy')
z_space, Fz_TM = np.load('npy_data/Fz_TM_sm.npy')
z_space, Fz_TE_mm = np.load('npy_data/Fz_TE_mm.npy')
z_space, Fz_TM_mm = np.load('npy_data/Fz_TM_mm.npy')


fig, ax = plt.subplots(figsize=[10,6])
#plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 12})
plt.title('$V = $ %.3f, $\lambda=$%.0f nm, $R_c = $%.0f nm, $R_p = $%.0f nm, $P_{laser}=$%.1f mW\n$\epsilon_{out}= $%.2f, $\epsilon_{fiber}= $%.2f, $\epsilon_{particle}= $%.2f,' % (V, lammmm, fiber_radius/2/np.pi*lammmm, R_particle/2/np.pi*lammmm, P_laser*1e3, eps_out, eps_in, eps_particle))
plt.plot(z_space/lam, Fz_TE * 1e12, label='TE sm', color='b', alpha=0.7)
plt.plot(z_space/lam, Fz_TM * 1e12, label='TM sm', color='r', alpha=0.7)
plt.plot(z_space/lam, Fz_TE_mm * 1e12, '--',label='TE mm', color='b')
plt.plot(z_space/lam, Fz_TM_mm * 1e12, '--',label='TM mm', color='r')
plt.legend(shadow=True)
plt.xlabel('Distance between particles $\Delta z / \lambda$')
plt.ylabel('Longitudinal force $F_z$, pN')
plt.grid()

axins = zoomed_inset_axes(ax, 7, loc=4)
axins.plot(z_space/lam, Fz_TE * 1e12, label='TE', color='b', alpha=0.7)
axins.plot(z_space/lam, Fz_TM * 1e12, label='TM', color='r', alpha=0.7)
plt.plot(z_space/lam, Fz_TE_mm * 1e12, '--',label='TE mm', color='b')
plt.plot(z_space/lam, Fz_TM_mm * 1e12, '--',label='TM mm', color='r')
plt.xticks(visible=False)
plt.ylabel('$F_z$, pN')

x1, x2, y1, y2 = 10, 11, -1.25, 1.24

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
plt.grid()

plt.show()

