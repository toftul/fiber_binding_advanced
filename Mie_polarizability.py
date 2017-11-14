"""
@author: ivan

Last change: 08.11.2017
"""

import numpy as np
import scipy.special as sp

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
    """Calculates the exact polarizability of a sphere
    
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
    return(1.5j * a1(m, x) * (a / x)**3)
