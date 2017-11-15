"""
@author: ivan

Last change: 07.11.2017
"""


import numpy as np
import scipy.special as sp

# see theory from Craig F Bohren & Donald R Huffman
# Absorbtion and scattering of light by Small Particles
# Ch 8 par 4


def Es(r, phi, z, k, R, m, E0, nmin, nmax, case):
    """Scattered plane wave el-field from an infinite cylinder 
       in cylindrical coodinates 
    Case I: a Transverse Magnetic (TM) mode. 
        The magnetic field of the incident wave is 
        perpendicular to the cylinder axis.
    Case II: a Transverse Electric (TE) mode.
        The electric field is perpendicular 
        to the cylinder axis.

    Only for normal incident light (zeta = pi/2) 
    which simplifies to 
    h = - k cos zeta = 0, l = k
    a_nI = 0, b_nII = 0

    Parameters
    ----------
        r, phi, z : float
            reciever coordinates;
        k : float
            k-vector value, 2pi/\lambda_0 = \omega/c;
        R : float
            cylynder radius;
        m : float (complex?)
            refractive index of the cylinder relative 
            to that of the surrounding medium;
        E0 : float
            the el-field amplitude of incident wave;
        nmin, nmax : int
            min and max indexes in the sum;
        case : int
            1 --- case I;
            2 --- case II
    
    Returns
    -------
        Es : numpy array
            vector of scattered field
            in cylindrical coordinates
    
    """
    l = k
    # h = 0
    rho = r * l
    x = k * R
    mx = m * x

    Es = np.zeros(3, dtype=complex)
    for n in range(nmin, nmax + 1):
        Zn = sp.hankel1(n, rho)

        Jn_x = sp.jv(n, x)
        Jn_mx = sp.jv(n, mx)
        Jnp_x = sp.jvp(n, x)
        Jnp_mx = sp.jvp(n, mx)
        H1np = sp.h1vp(n, x)
        H1n = sp.hankel1(n, x)
        # Case I
        if case == 1:
            b_nI = (Jn_mx * Jnp_x - m * Jnp_mx * Jn_x) / \
                   (Jn_mx * H1np - m * Jnp_mx * H1n)
            Es -= np.exp(1j * (n * phi - np.pi / 2)) * \
                b_nI / k * np.array([0, 0, l * Zn])
        # Case II
        if case == 2:
            Znp = sp.h1vp(n, rho)
            a_nII = (m * Jnp_x * Jn_mx - Jn_x * Jnp_mx) / \
                    (m * Jn_mx * H1np - Jnp_mx * H1n)
            Es += np.exp(1j * (n * phi - np.pi / 2)) * \
                a_nII * np.array([1j * n * Zn / rho, -Znp, 0])

    return(Es * E0)
