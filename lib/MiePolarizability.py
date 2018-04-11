"""
@author: ivan

Last change: 08.11.2017
"""

import numpy as np
import scipy.special as sp
import lib.const as const

# Polarizability coef
# psi
def ricc1_psi(x):
    return(x * sp.spherical_jn(1, x, 0))


# psi'
def ricc1_psi_ch(x):
    return(sp.spherical_jn(1, x, 0) + x * sp.spherical_jn(1, x, 1))


# xi
def ricc1_xi(x):
    return(x * (sp.spherical_jn(1, x, 0) + 1j * sp.spherical_yn(1, x, 0)))

# xi'


def ricc1_xi_ch(x):
    return((sp.spherical_jn(1, x, 0) +
            1j * sp.spherical_yn(1, x, 0)) +
           x * (sp.spherical_jn(1, x, 1) +
                1j * sp.spherical_yn(1, x, 1)))


# ## Mie theory
# m = sqrt(epsilon_particle) / sqrt(epsilon_medium)
# x = k a
def a1(m, x):
    up = m * ricc1_psi(m * x) * ricc1_psi_ch(x) - \
        ricc1_psi(x) * ricc1_psi_ch(m * x)
    bot = m * ricc1_psi(m * x) * ricc1_xi_ch(x) - \
        ricc1_xi(x) * ricc1_psi_ch(m * x)
    return (up / bot)


def polarizability(k, a, epsilon_p, epsilon_m):
    """Calculates the exact polarizability of a sphere (SI units)
    
    Parameters
    ----------
        k : float
            wavelength in vacuum,
            k = omega/c;
        a : float
            sphere radius;
        epsilon_p, epsilon_m : complex
            epsilon of particle and media
    
    Returns
    -------
        alpha : complex
            polarizability of a sphere
    """
    x = np.sqrt(epsilon_m) * k * a
    m = np.sqrt(epsilon_p / epsilon_m + 0j)
    return(4 * np.pi * const.epsilon0 * 1.5j * a1(m, x) * (a / x)**3)
    

def polarizability_dipole(a, epsilon_p, epsilon_m):
    """Calculates the exact polarizability of a sphere (SI units)
    
    Parameters
    ----------
        a : float
            sphere radius;
        epsilon_p, epsilon_m : complex
            epsilon of particle and media
    
    Returns
    -------
        alpha : complex
            polarizability of a sphere in a dipole approx
    """
    return(4 * np.pi * const.epsilon0 * a**3 * 
           (epsilon_p - epsilon_m) / (epsilon_p + 2 * epsilon_m))
    
    
def polarizability_dipole_rad_corr(k, a, epsilon_p, epsilon_m):
    """Calculates the exact polarizability of a sphere (SI units)
    with radiation correnctions
    
    Parameters
    ----------
        a : float
            sphere radius;
        epsilon_p, epsilon_m : complex
            epsilon of particle and media
    
    Returns
    -------
        alpha : complex
            polarizability of a sphere in a dipole approx 
            with radiation corrections
    """
    alpha0 = 4 * np.pi * const.epsilon0 * a**3 * \
             (epsilon_p - epsilon_m) / (epsilon_p + 2 * epsilon_m)
    return(alpha0 / (1 - 1j * alpha0 * k*k*k/(6*np.pi)))

# %%
    
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.rcParams.update({'font.size': 14})
## plt.style.use('ggplot')
#lam_space = np.linspace(200, 1200, 400)
#a_nm = 120
#k = 2 * np.pi / lam_space * a_nm
#a = 1
#eps_p = 2.5
#eps_m = 1.77
#
#alpa_mie = polarizability(k, a, eps_p, eps_m) / (4*np.pi)
#alpa_dipole = polarizability_dipole(a, eps_p, eps_m) / (4*np.pi)
#alpa_dipole_rad_corr = polarizability_dipole_rad_corr(k, a, eps_p, eps_m) / (4*np.pi)
#
#plt.figure(figsize=(7,6))
#plt.title('$R_{p} = $%.1f nm, $\epsilon_p = $%.2f, $\epsilon_m = $%.2f, ' % (a_nm, eps_p, eps_m))
#plt.plot(lam_space, alpa_mie.real, 'C0', label='Mie theory (exact)')
#plt.plot(lam_space, alpa_mie.imag, 'C0--')
#plt.plot(lam_space, alpa_dipole_rad_corr.real, 'C1', label='Radiation correction')
#plt.plot(lam_space, alpa_dipole_rad_corr.imag, 'C1--')
#plt.plot(lam_space, alpa_dipole.real * k/k, 'C3', label='Dipole model')
#plt.plot(lam_space, alpa_dipole.imag * k/k, 'C3--')
#plt.plot(np.nan, np.nan, 'k', label='Re')
#plt.plot(np.nan, np.nan, 'k--', label='Im')
#plt.xlabel(r'wavelength $\lambda$, nm')
#plt.ylabel(r'Polarizability $\alpha / 4\pi\epsilon_0 R_p^3$')
#plt.xlim(np.min(lam_space), np.max(lam_space))
#plt.legend(fontsize=12)
#plt.grid()
#plt.savefig('polarizabllity.pdf')
#plt.show()

