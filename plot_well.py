#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:53:15 2018

@author: ivan
"""

import numpy as np
import const
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import gridspec
#from basic_units import radians
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import interp2d
#from scipy.interpolate import RectBivariateSpline
#plt.style.use('ggplot')

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

U_data = np.zeros([len(z_space), len(phi_space)])
for i, z_i in enumerate(z_space):
    for j, phi_j in enumerate(phi_space):
        U_data[i, j] = U_(phi_j, z_i)
        
U_data -= np.min(U_data) 
kT = const.k_B * 300 * 1e9
U_data /= kT
U_data += 15
#U_data /= np.max(U_data)
#U_data += 0.5
#U_data *= 2

# simple plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_ylabel('Distance to the first particle, $\Delta z / \lambda$')
ax.set_xlabel('Twisting angle, $\phi$, rad')
ax.set_zlabel(r'Trapping potential, $U/kT$')
PHI = phi_space
Z = z_space
PHI, Z = np.meshgrid(PHI, Z)
surf1 = ax.plot_surface(PHI, Z/(2*np.pi), U_data, 
                        cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.9)
plt.show()

# %%

x_space = np.zeros([len(z_space), len(phi_space)])
y_space = np.zeros([len(z_space), len(phi_space)])
zz_space = np.zeros([len(z_space), len(phi_space)])
for i, z_i in enumerate(z_space):
    for j, phi_j in enumerate(phi_space):
        x_space[i, j] = U_data[i, j] * np.cos(phi_j)
        y_space[i, j] = U_data[i, j] * np.sin(phi_j)
        zz_space[i, j] = z_i
        

phi_space_cyl = np.linspace(0, 2*np.pi)
x_cyl = np.zeros([len(z_space), len(phi_space_cyl)])
y_cyl = np.zeros([len(z_space), len(phi_space_cyl)])
zz_cyl = np.zeros([len(z_space), len(phi_space_cyl)])
for i, z_i in enumerate(z_space):
    for j, phi_j in enumerate(phi_space_cyl):
        r_cyl = np.min(U_data) * .9
        x_cyl[i, j] = r_cyl * np.cos(phi_j)
        y_cyl[i, j] = r_cyl * np.sin(phi_j)
        zz_cyl[i, j] = z_i
        

###### PLOTTING ######
        
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_ylabel('Distance to the first particle, $\Delta z / \lambda$')

# Make data.
PHI = phi_space
Z = z_space
PHI, Z = np.meshgrid(PHI, Z)


color_dimension = np.sqrt(x_space * x_space + y_space * y_space)
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

# Plot the surface.
x_space = - x_space
#surf = ax.plot_surface(PHI, Z/(2*np.pi), U_data, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
surf1 = ax.plot_surface(y_space, zz_space/(2*np.pi), x_space, 
                       facecolors=fcolors, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.5)
cset = ax.contourf(y_space, zz_space/(2*np.pi), x_space, zdir='z', offset=-r_cyl * 1.5, facecolors=fcolors, alpha=0.3)
cset = ax.contour(y_space, zz_space/(2*np.pi), x_space, zdir='x', 
                  offset=-np.max(U_data)-0.5, facecolors=fcolors, alpha=0.8, levels=[0])
cset = ax.contour(y_space, zz_space/(2*np.pi), x_space, zdir='x', 
                  offset=0, facecolors=fcolors, alpha=0.8, levels=[0])
# phi dependence
levels_pos = np.array([3.3, 5.5])
levels_colors = ['m', 'r']
for i, lev in enumerate(levels_pos):
    cset = ax.contour(y_space, zz_space/(2*np.pi), x_space, zdir='y', 
                  offset=np.min(zz_space)/(2*np.pi) - 0.3, colors=levels_colors[i], alpha=0.8, levels=[lev])
    cset = ax.contour(y_space, zz_space/(2*np.pi), x_space, zdir='y', 
                  offset=lev, colors=levels_colors[i], alpha=0.8, levels=[lev], linewidths=3)
surf2 = ax.plot_surface(y_cyl, zz_cyl/(2*np.pi), x_cyl, linewidth=0, antialiased=False, alpha=0.9)

# add polar grid
y0 = np.min(zz_space)/(2*np.pi) - 0.3
rmod = 110
for theta in np.arange(-90, 91, 15)*np.pi/180:
    arrows_x = [0, np.sin(theta)*rmod]
    arrows_y = [y0, y0]
    arrows_z = [0, np.cos(theta)*rmod]
    ax.plot(arrows_x, arrows_y, arrows_z, color = 'gray', linewidth=0.5)

for rmod in np.arange(0.5, rmod+0.1, int(rmod/8)):
    theta = - np.pi/2
    arrows_x = [np.sin(theta)*rmod]
    arrows_y = [y0]
    arrows_z = [np.cos(theta)*rmod]
    dtheta = 1
    for theta in np.arange(-90+dtheta, 90.1, dtheta)*np.pi/180:
        arrows_x = np.append(arrows_x, np.sin(theta)*rmod)
        arrows_y = np.append(arrows_y, y0)
        arrows_z = np.append(arrows_z, np.cos(theta)*rmod)
    ax.plot(arrows_x, arrows_y, arrows_z, color = 'gray', linewidth=0.5)

# add grid on the back
x0 = -np.max(U_data)-0.5
for zb_i in np.arange(0, np.max(U_data)*1.1, np.max(U_data)*1.1 / 6):
    arrows_x = [x0, x0]
    arrows_y = [np.min(zz_space)/(2*np.pi)*.9, np.max(zz_space)/(2*np.pi)]
    arrows_z = [zb_i, zb_i]
    ax.plot(arrows_x, arrows_y, arrows_z, color = 'gray', linewidth=0.5)

# Customize axis.
ax.set_xlim(- np.max(U_data), np.max(U_data))
ax.set_zlim(-r_cyl * 1.5, np.max(U_data))
ax.set_ylim(np.min(zz_space)/(2*np.pi) -.3, np.max(zz_space)/(2*np.pi))

ax.set_xticks([], minor=False)
ax.xaxis.grid(True, which='minor')

ax.set_zticks([], minor=False)
ax.zaxis.grid(True, which='minor')

# add text
ax.text(-np.max(U_data), np.max(U_data)/2, np.max(U_data)*.4, r'$U(\phi=0, \Delta z)$')
ax.text(np.max(U_data)*0.8, y0*0.7, np.max(U_data)*1.1, r'$U(\phi, \Delta z=$const)')

# Add a color bar which maps values to colors.
cbar = fig.colorbar(m, shrink=0.6, aspect=15, ticks=[np.min(U_data), np.max(U_data)])
cbar.ax.set_yticklabels([r'$U_{\min}$', r'$U_{\max}$'])

ax.set_title(r'Trapping potential, $U(\rho = $const$, \phi, \Delta z)$')

plt.show()


#plt.plot(z_space/(2*np.pi), Fz_interpolated(0, z_space))
#plt.show()