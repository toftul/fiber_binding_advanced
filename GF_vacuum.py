"""
@author: ivan

Last change: 20.10.2017
"""

import numpy as np

def GF_vac(R, k):
    """Green's function of Maxwell eq. in a free space

    Parameters
    ----------
        R : numpy 1D array 1x3;
        k : float
            k-vector value, 2pi/\lambda_0 = \omega/c;

    Returns
    -------
        G : complex numpy 2D array 3x3

    Details
    -------
        R = r1 - r2;
        r1 -- source position;
        r2 -- reciever postion;
        omega -- radiation frequency

    """
    G = np.zeros([3, 3], dtype=complex)

    if np.linalg.norm(R) != 0:
        Rmod = np.linalg.norm(R)
        kR = k * Rmod
        EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
        CONST1 = (1 + (1j * kR - 1) / kR**2)
        CONST2 = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2)
        # return Green function
        G = EXPikR4piR * (CONST1 * np.identity(3) +
                          CONST2 * np.outer(R, R))
    else:
        print("Error in GF_vac. R = 0 -> devision by zero!")

    return G


def GF_vac_pol(r1_pol, r2_pol, k):
    """Green's function of Maxwell eq. in a free space
    in polar coordinates
        
        G_pol = Q^T G_car Q

    For details see https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
        r1_pol, r2_pol : numpy 1D array 1x3;
            r_pol = (rho, theta, z)
            r1 -- source position;
            r2 -- reciever postion;
        k : float
            k-vector value, 2pi/\lambda_0 = \omega/c;

    Returns
    -------
        G : complex numpy 2D array 3x3

    Details
    -------
        R = r1 - r2;
        r1 -- source position;
        r2 -- reciever postion;
        omega -- radiation frequency

    """
    r1 = np.array([r1_pol[0] * np.cos(r1_pol[1]),
                   r1_pol[0] * np.sin(r1_pol[1]),
                   r1_pol[2]])
    r2 = np.array([r2_pol[0] * np.cos(r2_pol[1]),
                   r2_pol[0] * np.sin(r2_pol[1]),
                   r2_pol[2]])
    
    G_car = GF_vac(r1 - r2, k)

    Q1 = np.array([[np.cos(r1_pol[1]), np.sin(r1_pol[1]), 0],
                  [-np.sin(r1_pol[1]), np.cos(r1_pol[1]), 0],
                  [0, 0, 1]])
    
    Q2 = np.array([[np.cos(r2_pol[1]), np.sin(r2_pol[1]), 0],
                  [-np.sin(r2_pol[1]), np.cos(r2_pol[1]), 0],
                  [0, 0, 1]])

    return Q1.transpose() @ G_car @ Q2
