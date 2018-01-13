import numpy as np
import const
import GF_fiber as gff
import GF_vacuum as gfv
import Mie_scat_cyl
import Mie_polarizability as mie_alpha
import matplotlib.pyplot as plt

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


def E0_sum(r, k, fiber_radius, eps_out, eps_in, E0_mod, nmin_sc, nmax_sc, case):
    """Calculates external electric field in polar coodrinates
        E0 = E_inc + E_sc

    Case I: a Transverse Magnetic (TM) mode. 
        The magnetic field of the incident wave is 
        perpendicular to the cylinder axis.
    Case II: a Transverse Electric (TE) mode.
        The electric field is perpendicular 
        to the cylinder axis.

    Parameters
    ----------
        r : numpy array
            r = (rho, theta, z)

    Returns
    -------
        E0 : numpy complex array
    """

    # refractive index of the cylinder relative 
    # to that of the surrounding medium
    m = np.sqrt(eps_in / eps_out)
    E0 = Mie_scat_cyl.Es(r[0], r[1], r[2], k, fiber_radius,
                         m, E0_mod, nmin_sc, nmax_sc, case)

    r_car = pol2cart(r)
    kvec_car = np.array([-k, 0, 0])  # normal incidence
    exp_factor = np.exp(1j * np.dot(kvec_car, r_car))
    if case == 1:
        Einc_car = np.array([0, 0, E0_mod], dtype=complex) * exp_factor
        # Ez is the same in pol and in cart coordinates
        E0 += Einc_car
    elif case == 2:
        Einc_car = np.array([0, E0_mod, 0], dtype=complex) * exp_factor
        E0 += vec_cart2pol(r_car, Einc_car)

    return(E0)



def dipole_moment(i, r1, r2, R_particle, eps_particle,
                  k, eps_out, eps_in, fiber_radius,
                  nmin, nmax, kzimax,
                  E0_mod, nmin_sc, nmax_sc, case):
    """Calculates dipole moment

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

    alpha0 = mie_alpha.polarizability(k, R_particle, eps_particle, eps_out)

    k2_eps0 = k**2 / const.epsilon0
    Gsjj = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                  rj, rj, nmin, nmax, kzimax)
    Gsii = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                  ri, ri, nmin, nmax, kzimax)
    G0ij = gfv.GF_vac_pol(ri, rj, k)
    G0ji = G0ij.transpose()

    Gsij = gff.GF_pol(k, eps_out, eps_in, fiber_radius, ri, rj, nmin, nmax, kzimax)

    Gsji = Gsij.transpose()
    Gsji[0, 1] = gff.GF_pol_ij(k, eps_out, eps_in, fiber_radius, rj, ri, nmin, nmax,
                               0, 1, kzimax)[0]
    Gsji[1, 1] = gff.GF_pol_ij(k, eps_out, eps_in, fiber_radius, rj, ri, nmin, nmax,
                               1, 1, kzimax)[0]
    Gsji[1, 0] = gff.GF_pol_ij(k, eps_out, eps_in, fiber_radius, rj, ri, nmin, nmax,
                               1, 0, kzimax)[0]

    Gij = G0ij + Gsij
    Gji = G0ji + Gsji

    alpha_sj = alpha0 * np.linalg.inv(np.eye(3) - alpha0 * k2_eps0 * Gsjj)
    alpha_si = alpha0 * np.linalg.inv(np.eye(3) - alpha0 * k2_eps0 * Gsii)

    E0j = E0_sum(rj, k, fiber_radius, eps_out, eps_in,
                 E0_mod, nmin_sc, nmax_sc, case)
    E0i = E0_sum(ri, k, fiber_radius, eps_out, eps_in,
                 E0_mod, nmin_sc, nmax_sc, case)

    num = np.dot(alpha_si, E0i + k2_eps0 * np.dot(Gij @ alpha_sj, E0j))
    dem = np.eye(3) - k2_eps0**2 * alpha_si @ Gij @ alpha_sj @ Gji

    pi = np.dot(np.linalg.inv(dem), num)
    return(pi)


def total_loc_efield(i, r1, r2, k, case, nmin, nmax, kzimax,
                     fiber_radius, eps_out, eps_in, E0_mod, nmin_sc, nmax_sc,
                     R_particle, eps_particle):
    """Calculates E_loc in 
        F = 1/2 Re p^* nabla E_loc

    Case I: a Transverse Magnetic (TM) mode. 
        The magnetic field of the incident wave is 
        perpendicular to the cylinder axis.
    Case II: a Transverse Electric (TE) mode.
        The electric field is perpendicular 
        to the cylinder axis.
        
    Parameters
    ----------
        i : int
            considered particle: r1 or r2
            i = 1, 2
        r1, r2 : numpy array
            r = (rho, theta, z)
        E0 : float
            the magnitude of the incident wave
        case : int
            case = 1, 2
            
    Returns
    -------
        E : compex numpy array
        E = (Erho, Etheta, Ez)
    """
    if i == 1:
        ri = r1
        rj = r2
        j = 2
    elif i == 2:
        ri = r2
        rj = r1
        j = 1
    else:
        ri = np.array([0, 0, 0])
        rj = np.array([0, 0, 0])
        j = 0
        print('ERROR: i is out of range!')

    k2_eps0 = k**2 / const.epsilon0
    E0i = E0_sum(ri, k, fiber_radius, eps_out, eps_in,
                 E0_mod, nmin_sc, nmax_sc, case)

    Gsii = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                      ri, ri, nmin, nmax, kzimax)
    G0ij = gfv.GF_vac_pol(ri, rj, k)

    Gsij = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                      ri, rj, nmin, nmax, kzimax)
    Gij = G0ij + Gsij

    pi = dipole_moment(i, r1, r2, R_particle, eps_particle, k, eps_out, eps_in,
                       fiber_radius, nmin, nmax, kzimax,
                       E0_mod, nmin_sc, nmax_sc, case)
    pj = dipole_moment(j, r1, r2, R_particle, eps_particle, k, eps_out, eps_in,
                       fiber_radius, nmin, nmax, kzimax,
                       E0_mod, nmin_sc, nmax_sc, case)
    return(E0i + k2_eps0 * (np.dot(Gij, pj) + np.dot(Gsii, pi)))


def force_12(alpha, r1, r2, R_particle, eps_particle, k, eps_out, eps_in,
             fiber_radius, nmin, nmax, kzimax, E0_mod, nmin_sc, nmax_sc, case):
    """Calculates force acting on 1st particle from 2nd
        F12 = 1/2 Re p1* grad E_loc

    Parameters
    ----------
        alpha : int
            considered component of the force
            alpha = 0, 1, 2
            F = (Frho, Ftheta, Fz)
        r1, r2 : numpy array
            positions of particles
            r = (rho, theta, z)
        R_particle : float
            particle radius, R1 = R2
        eps_particle : complex
            epsilon of the particle
        k : float
            incident wave vector, k = omega/c
        eps_out : float
            epsilon of media around fiber and particles
        eps_in : float 
            epsilon of fiber
        fiber_radius : float
        nmin, nmax : int 
            indices in sum in fiber Green function
            WARNING: large values may strongly increase computation time
        kzimax : float
            integration limit in fiber Green function
        E0_mod : float
            amplitude of incident wave
        nmin_sc, nmax_sc : int
            indices in sum in scatterred field
        case : int
            case = 1, 2
            Case I: a Transverse Magnetic (TM) mode. 
                The magnetic field of the incident wave is 
                perpendicular to the cylinder axis.
            Case II: a Transverse Electric (TE) mode.
                The electric field is perpendicular 
                to the cylinder axis.

    Returns
    -------
        F_alpha : float
            alpha's component for F12 

    """

    dr = 1 / k * 1e-5
    dz = dr
    dtheta = 1e-5

    p1 = dipole_moment(1, r1, r2, R_particle, eps_particle, k, eps_out, eps_in,
                       fiber_radius, nmin, nmax, kzimax,
                       E0_mod, nmin_sc, nmax_sc, case)
    p1c = p1.conjugate()

    # Fr
    if alpha == 0:
        r1plusdr = r1 + np.array([dr, 0, 0])
        r1minusdr = r1 - np.array([dr, 0, 0])
        Eplusr = total_loc_efield(1, r1plusdr, r2, k, case, nmin, nmax, kzimax,
                                  fiber_radius, eps_out, eps_in, E0_mod,
                                  nmin_sc, nmax_sc, R_particle, eps_particle)
        Eminusr = total_loc_efield(1, r1minusdr, r2, k, case, nmin, nmax, kzimax,
                                   fiber_radius, eps_out, eps_in, E0_mod,
                                   nmin_sc, nmax_sc, R_particle, eps_particle)
        grad_r = (Eplusr - Eminusr) / (2 * dr)

        return(0.5 * np.dot(p1c, grad_r).real)
    # Ftheta
    elif alpha == 1:
        r1plusdtheta = r1 + np.array([0, dtheta, 0])
        r1minusdtheta = r1 - np.array([0, dtheta, 0])

        Eplustheta = total_loc_efield(1, r1plusdtheta, r2, k, case, nmin, nmax, kzimax,
                                      fiber_radius, eps_out, eps_in, E0_mod,
                                      nmin_sc, nmax_sc, R_particle, eps_particle)
        Eminustheta = total_loc_efield(1, r1minusdtheta, r2, k, case, nmin, nmax, kzimax,
                                       fiber_radius, eps_out, eps_in, E0_mod,
                                       nmin_sc, nmax_sc, R_particle, eps_particle)
        grad_theta = (Eplustheta - Eminustheta) / (r1[0] * 2 * dtheta)

        return(0.5 * np.dot(p1c, grad_theta).real)
    # Fz
    elif alpha == 2:
        r1plusdz = r1 + np.array([0, 0, dz])
        r1minusdz = r1 - np.array([0, 0, dz])

        Eplusz = total_loc_efield(1, r1plusdz, r2, k, case, nmin, nmax, kzimax,
                                  fiber_radius, eps_out, eps_in, E0_mod,
                                  nmin_sc, nmax_sc, R_particle, eps_particle)
        Eminusz = total_loc_efield(1, r1minusdz, r2, k, case, nmin, nmax, kzimax,
                                   fiber_radius, eps_out, eps_in, E0_mod,
                                   nmin_sc, nmax_sc, R_particle, eps_particle)
        grad_z = (Eplusz - Eminusz) / (2 * dz)

        return(0.5 * np.dot(p1c, grad_z).real)
    else:
        print('alpha is out of range!')
        return(0)




lam = 550*1e-9
k = 2 * np.pi / lam
R_particle = 100*1e-9
fiber_radius = 100*1e-9

eps_particle = 2.5
eps_out = 1
eps_in = 2.09

nmin = 0
nmax = 1
kzimax = 15*k


P_laser = 100e-3  # [W]
R_focus = 1e-6  # [m]
Intensity = P_laser / (np.pi * R_focus**2)  # [W/m^2]
E0_mod = np.sqrt(0.5 * const.Z0 * Intensity)  # [V/m]

# E0_mod *= lam / (2 * np.pi) # [V], [k] = [1]

nmin_sc = -60
nmax_sc = 60
case = 1
r1 = np.array([fiber_radius + R_particle, 0, 0])
r2 = np.array([fiber_radius + R_particle, 0, 2*R_particle])


z_space = np.linspace(2 * R_particle, 7 * lam, 55)
Fz = np.zeros(len(z_space))
for i, zz in enumerate(z_space):
    print('step = ', i)
    r2 = np.array([fiber_radius + R_particle, 0, zz])
    Fz[i] = force_12(2, r1, r2, R_particle, eps_particle, k, eps_out, eps_in, 
              fiber_radius, nmin, nmax, kzimax, E0_mod, nmin_sc, nmax_sc, case)
    
np.save('Fz_onemode_sc', (z_space, Fz))
#z_space, Fz = np.load('Fz_onemode_sc.npy')

#plt.plot(z_space/lam, Fz)
#plt.grid()
#plt.show()

