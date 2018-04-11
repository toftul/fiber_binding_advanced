#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:27:15 2018

@author: ivan
"""

import numpy as np
import lib.const as const
#import GF_fiber_cython as gff
import lib.GFfiber as gff
import lib.GFvacuum as gfv
import lib.MiePolarizability as mie_alpha
import matplotlib.pyplot as plt

# plt.style.use('ggplot')

# %%
# FUNCTIONS

def cart2pol(r):
    rho = np.sqrt(r[0]**2 + r[1]**2)
    phi = np.arctan2(r[1], r[0])
    return(np.array([rho, phi, r[2]]))


def pol2cart(r):
    x = r[0] * np.cos(r[1])
    y = r[0] * np.sin(r[1])
    return(np.array([x, y, r[2]]))

def vec_cart2pol(r, A):
    theta = np.arctan2(r[1], r[0])
    Q = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return(np.dot(Q, A))


def alphaEff(i, r1, r2, R_particle, eps_particle,
             k, eps_out, eps_in, fiber_radius,
             nmin, nmax, kzimax):
    """Calculates dipole moment
    It is assumed that rho1 = rho2

    Parameters
    ----------
        i : int
            1, 2 -- number of considered particle,
            refers to r1 or r2 accordingly;
        r1, r2 : numpy vecvors
            positions of particles in polar coordinates;
            r = (rho, theta, z)
        R_particle : float
            particles' radius, assuming R1 = R2
        eps_particle : complex
            particle's epsilon (depends on k)
    
    Returns
    -------
        pi : numpy array
            dipole moment
    """

    if i == 1:
        ri = r1
        rj = r2
    elif i == 2:
        ri = r2
        rj = r1
    else:
        ri = np.array([0, 0, 0])
        rj = np.array([0, 0, 0])
        print('ERROR: i is out of range!')

    I = np.eye(3)
    #alpha0 = mie_alpha.polarizability_dipole(R_particle, eps_particle, eps_out)
    alpha0 = mie_alpha.polarizability(k, R_particle, eps_particle, eps_out)

    k2_eps0 = k**2 / const.epsilon0
    Gsjj = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                  rj, rj, nmin, nmax, kzimax)
    Gsii = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                  ri, ri, nmin, nmax, kzimax)
    G0ij = gfv.GF_vac_pol(ri, rj, k, eps_out)
    G0ji = G0ij.transpose()

    Gsij = gff.GF_pol(k, eps_out, eps_in, fiber_radius, ri, rj, nmin, nmax, kzimax)
    
    if ri[1] - rj[0] == 0.0:
        Gsji = Gsij
        Gsji[0, 2] = - Gsij[0, 2]
        Gsji[2, 0] = Gsij[0, 2]
    else:
        Gsji = gff.GF_pol(k, eps_out, eps_in, fiber_radius, rj, ri, nmin, nmax, kzimax)
    

    Gij = G0ij + Gsij
    Gji = G0ji + Gsji

    alpha_sj = alpha0 * np.linalg.inv(I - alpha0 * k2_eps0 * Gsjj)
    alpha_si = alpha0 * np.linalg.inv(I - alpha0 * k2_eps0 * Gsii)

    num = np.dot(alpha_si, I + k2_eps0 * (Gij @ alpha_sj))
    dem = I - k2_eps0**2 * alpha_si @ Gij @ alpha_sj @ Gji

    return(np.dot(np.linalg.inv(dem), num))
    
    
def alphaEffVacuum(i, r1, r2, R_particle, eps_particle,
             k, eps_out, eps_in, fiber_radius,
             nmin, nmax, kzimax):
    """Calculates dipole moment
    It is assumed that rho1 = rho2

    Parameters
    ----------
        i : int
            1, 2 -- number of considered particle,
            refers to r1 or r2 accordingly;
        r1, r2 : numpy vecvors
            positions of particles in polar coordinates;
            r = (rho, theta, z)
        R_particle : float
            particles' radius, assuming R1 = R2
        eps_particle : complex
            particle's epsilon (depends on k)
    
    Returns
    -------
        pi : numpy array
            dipole moment
    """

    if i == 1:
        ri = r1
        rj = r2
    elif i == 2:
        ri = r2
        rj = r1
    else:
        ri = np.array([0, 0, 0])
        rj = np.array([0, 0, 0])
        print('ERROR: i is out of range!')

    I = np.eye(3)
    #alpha0 = mie_alpha.polarizability_dipole(R_particle, eps_particle, eps_out)
    alpha0 = mie_alpha.polarizability(k, R_particle, eps_particle, eps_out)

    k2_eps0 = k**2 / const.epsilon0
    Gsjj = 0.#gff.GF_pol(k, eps_out, eps_in, fiber_radius,
             #     rj, rj, nmin, nmax, kzimax)
    Gsii = 0.#gff.GF_pol(k, eps_out, eps_in, fiber_radius,
             #     ri, ri, nmin, nmax, kzimax)
    G0ij = gfv.GF_vac_pol(ri, rj, k, eps_out)
    G0ji = G0ij.transpose()

    Gsij = 0.#gff.GF_pol(k, eps_out, eps_in, fiber_radius, ri, rj, nmin, nmax, kzimax)
    
    if ri[1] - rj[0] == 0.0:
        Gsji = Gsij
        Gsji[0, 2] = - Gsij[0, 2]
        Gsji[2, 0] = Gsij[0, 2]
    else:
        Gsji = 0.#gff.GF_pol(k, eps_out, eps_in, fiber_radius, rj, ri, nmin, nmax, kzimax)
    

    Gij = G0ij + Gsij
    Gji = G0ji + Gsji

    alpha_sj = alpha0 * np.linalg.inv(I - alpha0 * k2_eps0 * Gsjj)
    alpha_si = alpha0 * np.linalg.inv(I - alpha0 * k2_eps0 * Gsii)

    num = np.dot(alpha_si, I + k2_eps0 * (Gij @ alpha_sj))
    dem = I - k2_eps0**2 * alpha_si @ Gij @ alpha_sj @ Gji

    return(np.dot(np.linalg.inv(dem), num))
    
# %%
# PARAMETERS

# sm -- single mode regime
# mm -- multy mode regime
regime = 'mm'

R_particleeee = 120.0  # [nm]
R_particle = 1.0


nmin = 0
nmax = 15
if regime == 'sm':
   fiber_radius = 130.0 / R_particleeee
elif regime == 'mm':
   fiber_radius = 495.0 / R_particleeee
else:
   fiber_radius = 1.0
   print('Check \'regime\' variable!')


eps_particle = 3.0  # 2.5
eps_out = 1.77
eps_in = 3.5  # 2.09


P_laser = 100e-3  # [W]
R_focus = 1e-6  # [m]
Intensity = P_laser / (np.pi * R_focus**2)  # [W/m^2]
E0_mod_real = np.sqrt(0.5 * const.Z0 * Intensity)  # [V/m]

# r =                rho,                       theta,     z
r1 = np.array([(fiber_radius + R_particle),     np.pi,     0.0             ])
r2 = np.array([(fiber_radius + R_particle),     np.pi,    10.0 * R_particle])


wlSpace = np.linspace(300, 1200, 250) / R_particleeee 
kSpace = 2*np.pi / wlSpace

alphaSpace = np.zeros([len(wlSpace), 3, 3], dtype=complex)
alphaSpaceVacuum = np.zeros([len(wlSpace), 3, 3], dtype=complex)
for j, k in enumerate(kSpace):
    V = k * fiber_radius * np.sqrt(eps_in - eps_out)
    kzimax = 5*k
    wl = wlSpace[j]
    print('V = %.3f   wl = %.1f   done: %.2f' % (V, wl*R_particleeee, j/len(kSpace)))
    alphaSpace[j] = alphaEff(1, r1, r2, R_particle, eps_particle,
                             k, eps_out, eps_in, fiber_radius,
                             nmin, nmax, kzimax)
    alphaSpaceVacuum[j] = alphaEffVacuum(1, r1, r2, R_particle, eps_particle,
                                         k, eps_out, eps_in, fiber_radius,
                                         nmin, nmax, kzimax)
    
# %%
i, j = 2, 2
alpha0 = 4*np.pi * const.epsilon0 * R_particle**3
alpha0 = mie_alpha.polarizability(2*np.pi/wlSpace, R_particle, eps_particle, eps_out)
name = r'$\alpha_{%d %d}$' %(i+1,j+1)
plt.title('$\Delta z/R_p = %.1f$, $R_f/R_p = %.1f$' % (r2[2] - r1[2], fiber_radius))
plt.xlabel('Wavelength, nm')
plt.ylabel(r'$\alpha / \alpha_{Mie}$')
plt.plot(wlSpace * R_particleeee, alphaSpace[:, i, j].real/alpha0.real, 'C0', label=r'$G_0 + G_s$')
plt.plot(wlSpace * R_particleeee, alphaSpace[:, i, j].imag/alpha0.imag, 'C0--')
plt.plot(wlSpace * R_particleeee, alphaSpaceVacuum[:, i, j].real/alpha0.real, 'C1', label=r'only $G_0$')
plt.plot(wlSpace * R_particleeee, alphaSpaceVacuum[:, i, j].imag/alpha0.imag, 'C1--')

plt.plot(np.nan, np.nan, 'k', label='Re'+name)
plt.plot(np.nan, np.nan, 'k--', label='Im'+name)
#
#plt.plot(wlSpace * R_particleeee, 
#         mie_alpha.polarizability(2*np.pi/wlSpace, R_particle, eps_particle, eps_out).real/alpha0, 
#         label=r'Re $\alpha$')
#plt.plot(wlSpace * R_particleeee, 
#         mie_alpha.polarizability(2*np.pi/wlSpace, R_particle, eps_particle, eps_out).imag/alpha0, 
#         label=r'Im $\alpha$')

plt.legend()
plt.grid()
if regime == 'sm':
   plt.savefig('results/alphaSpectrumSM.pdf')
elif regime == 'mm':
   plt.savefig('results/alphaSpectrumMM.pdf')
else:
   plt.show()

