import numpy as np


# Dimentionless Green func
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
