import numpy as np
import const
import GF_fiber as gff
import GF_vacuum as gfv
import Mie_scat_cyl
import Mie_polarizability as mie_alpha


def cart2pol(r):
    rho = np.sqrt(r[0]**2 + r[1]**2)
    phi = np.arctan2(r[1], r[0])
    return(np.array([rho, phi, r[2]]))


def pol2cart(r):
    x = r[0] * np.cos(r[1])
    y = r[0] * np.sin(r[1])
    return(np.array([x, y, r[2]]))


def alpha_eff(i, r1, r2, R_particle, eps_particle,
              k, eps_out, eps_in, rc, r1_vec,
              r2_vec, nmin, nmax, kzimax):
    """Calculates an effective polarizability: 
        pi = alpha_i E0

    Parameters
    ----------
        i : int
            1, 2 -- number of considered particle,
            refers to r1 or r2 accordingly;
        r1, r2 : numpy vecvors
            positions of particles;
        R_particle : float
            particles' radius, assuming R1 = R2
        eps_particle : complex
            particle's epsilon (depends on k)
    
    Returns
    -------
        alpha : numpy 3x3 tensor
            polirizability tensor
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

    alpha0 = mie_alpha.polarizability(k, R_particle, eps_particle, eps_out)

    k2_eps0 = k**2 / const.epsilon0
    Gsjj = gff.GF_car(k, eps_out, eps_in, rc,
                  rj, rj, nmin, nmax, kzimax)
    Gij = gfv.GF_vacuum(ri - rj, k) + \
          gff.GF_car(k, eps_out, eps_in, rc, ri, rj, nmin, nmax, kzimax)
    # Gij = Gji ???
    Gji = gfv.GF_vacuum(rj - ri, k) + \
          gff.GF_car(k, eps_out, eps_in, rc, rj, ri, nmin, nmax, kzimax)

    alpha_sj = alpha0 * np.inv(np.eye(3) - alpha0 * k2_eps0 * Gsjj)

    num = alpha0 * (np.eye(3) + k2_eps0 * np.dot(Gij, alpha_sj))
    dem = np.eye(3) - alpha0 * k2_eps0 * Gsjj - \
        alpha0 * k2_eps0**2 * np.dot(Gij, np.dot(alpha_sj, Gji))

    return(np.dot(np.linalg.inv(dem), num))


def total_loc_efield(i, r1, r2, k, E0, case):
    """Calculates E_loc in 
        F = 1/2 Re p^* nabla E_loc

        Case I: a Transverse Magnetic (TM) mode. 
            The magnetic field of the incident wave is 
            perpendicular to the cylinder axis.
        Case II: a Transverse Electric (TE) mode.
            The electric field is perpendicular 
            to the cylinder axis.
    """
    if case == 1:
        Einc = np.array(0, 0, 0)
    elif case = 2:
        Einc = np.array()
