import numpy as np
import lib.const as const
#import GF_fiber_cython as gff
import lib.GFfiber as gff
import lib.GFvacuum as gfv
import lib.MieScatCyl as Mie_scat_cyl
import lib.MiePolarizability as mie_alpha
import matplotlib.pyplot as plt
from matplotlib import gridspec
from basic_units import radians
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from joblib import Parallel, delayed
import time

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
    # Incidence is the following:
    #
    #       _
    #      / \     k
    #      \_/   <----
    #   -------------------------> x
    #
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
    G0ij = gfv.GF_vac_pol(ri, rj, k, eps_out)

    Gsij = gff.GF_pol(k, eps_out, eps_in, fiber_radius,
                      ri, rj, nmin, nmax, kzimax)
    Gij = G0ij + Gsij

    pi = dipole_moment(i, ri, rj, R_particle, eps_particle, k, eps_out, eps_in,
                       fiber_radius, nmin, nmax, kzimax,
                       E0_mod, nmin_sc, nmax_sc, case)
    pj = dipole_moment(j, ri, rj, R_particle, eps_particle, k, eps_out, eps_in,
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


def qPhiStable(z_eq):
    q = False
    return(q)

# single mode criteria
def VVV_q(wl, rho_c, epsilon_fiber, epsilon_m):
    V = 2*np.pi/wl * rho_c * np.sqrt(epsilon_fiber - epsilon_m)
    Vcr = 2.405
    lam_c = 1/Vcr * 2*np.pi * rho_c * np.sqrt(epsilon_fiber - epsilon_m)
    if V < Vcr:
        print('Single mode condition: PASSED!')
        #print('V/Vc = %.3f/2.405 < 1'% V)
    else:
        print('Single mode condition: FAILED!')
        #print('V/Vc = %.3f/2.405 > 1'% V)
    print('lambda critical = %.1f' % (lam_c * 1e9))


# %%
# PARAMETERS

# sm -- single mode regime
# mm -- multy mode regime
regime = 'sm'

eps_particle = 2.5
eps_out = 1.77
eps_in = 3.5  # 2.09

k = 1.0
lam = 2 * np.pi / k
lammmm_m = 530.0  # [nm] wavelength in media
lammmm = lammmm_m * np.sqrt(eps_out) # wavelength in vacuume 

R_particle = 120.0 / lammmm * 2 * np.pi / k

nmin = 0
nmax = 2
if regime == 'sm':
   #fiber_radius = 80.0 / lammmm * 2*np.pi / k
   fiber_radius = 130.0 / lammmm * 2*np.pi / k
elif regime == 'mm':
   nmin = 0
   nmax = 15
   fiber_radius = 495.0 / lammmm * 2*np.pi / k
else:
   fiber_radius = R_particle
   print('Check \'regime\' variable!')


#V = k * fiber_radius * np.sqrt(eps_in - eps_out)
#print('V = ', V)

kzimax = 5*k

P_laser = 100e-3  # [W]
R_focus = 1e-6  # [m]
Intensity = P_laser / (np.pi * R_focus**2)  # [W/m^2]
E0_mod_real = np.sqrt(0.5 * const.Z0 * Intensity)  # [V/m]

E0_mod = 1.0

nmin_sc = -60
nmax_sc = 60
# TM = 1, TE = 2
case = 2 
# r = (rho, theta, z)
r1 = np.array([(fiber_radius + R_particle), np.pi, 0])
r2 = np.array([(fiber_radius + R_particle), np.pi, 3.95 * lam])

if regime == 'sm':
   z_eq_space = np.array([3.95, 6.85, 9.0]) * lam
   z_eq_color_space = ['g', 'r', 'k']
elif regime == 'mm':
   z_eq_space = np.array([3.95, 6.85, 9.5]) * lam
   z_eq_color_space = ['g', 'r', 'k']
else:
   z_eq_space = np.array([3.95]) * lam
   z_eq_color_space = ['g']

# Creating grid for calculations
z_space = np.linspace(2 * R_particle, 12 * lam, 250)
phi_space = np.linspace(-np.pi/2, np.pi/2, 60) + np.pi
rho_space = np.linspace(fiber_radius + R_particle, 
                        fiber_radius + R_particle + 5*R_particle, 40)

r2_space = np.zeros([len(z_space), 3])
for i, zz in enumerate(z_space):
    r2_space[i] = np.array([r2[0], r2[1], zz])

r2_space_phi = np.zeros([len(z_eq_space), len(phi_space), 3])
for j, z_eq in enumerate(z_eq_space):
   for i, iphi in enumerate(phi_space):
       r2_space_phi[j, i] = np.array([r2[0], iphi, z_eq])
    

Frho = np.zeros(len(rho_space))  
Fphi = np.zeros([len(z_eq_space), len(phi_space)])
Fz = np.zeros(len(z_space))


# %%
### Fz calculation
start_time = time.time()
Fz = Parallel(n_jobs=4)(delayed(force_12)(2, r1, r2_space[i], R_particle, eps_particle, k, 
                                          eps_out, eps_in, fiber_radius, nmin, nmax, kzimax, 
                                          E0_mod, nmin_sc, nmax_sc, case) for i in range(len(z_space)))
Fz = np.asarray(Fz)
Fz = -Fz
# F to real units [N]
Fz = Fz * E0_mod_real**2 * const.epsilon00 * 2 * np.pi * lammmm*1e-9
ex_time = time.time() - start_time
print('Exucution time: %.1f s' % ex_time)

if regime == 'mm':
    np.save('npy_data/fzmm.npy', Fz)
elif regime == 'sm':
    np.save('npy_data/fzsm.npy', Fz)


# %%
### Fphi calculation
start_time = time.time()
for j in np.arange(len(z_eq_space)):
   Fphi_tmp = Parallel(n_jobs=4)(delayed(force_12)(1, r1, r2_space_phi[j, i], R_particle, eps_particle, k, 
                                                   eps_out, eps_in, fiber_radius, nmin, nmax, kzimax, 
                                                   E0_mod, nmin_sc, nmax_sc, case) for i in range(len(phi_space)))
   Fphi_tmp = np.asarray(Fphi_tmp)
   Fphi_tmp = - Fphi_tmp
   # F to real units [N]
   Fphi[j] = Fphi_tmp * E0_mod_real**2 * const.epsilon00 * 2 * np.pi * lammmm*1e-9
   
ex_time = time.time() - start_time
print('Exucution time: %.1f s' % ex_time)

if regime == 'mm':
    np.save('npy_data/fphimm.npy', Fphi)
elif regime == 'sm':
    np.save('npy_data/fphism.npy', Fphi)


# %%
### Fr calculation
#start_time = time.time()
#Frho = Parallel(n_jobs=4)(delayed(force_12)(0, r1, r2_space[i], R_particle, eps_particle, k, 
#                                          eps_out, eps_in, fiber_radius, nmin, nmax, kzimax, 
#                                          E0_mod, nmin_sc, nmax_sc, case) for i in range(len(rho_space)))
#Frho = np.asarray(Frho)
#Frho = -Frho
## F to real units [N]
#Frho = Frho * E0_mod_real**2 * const.epsilon00 * 2 * np.pi * lammmm*1e-9
#ex_time = time.time() - start_time
#print('Exucution time: %.1f s' % ex_time)


# %%
# Plot results

if regime == 'mm':
    Fz = np.load('npy_data/fzmm.npy')
    Fphi = np.load('npy_data/fphimm.npy')
elif regime == 'sm':
    Fz = np.load('npy_data/fzsm.npy')
    Fphi = np.load('npy_data/fphism.npy')


fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.8]) 
Fz_plt = fig.add_subplot(gs[0])
Fphi_plt = fig.add_subplot(gs[1])

# Fz plot
Fz_plt.plot(z_space/(2*np.pi), Fz*1e12, label='$F_z(\phi = 0, \Delta z)$', color='steelblue', zorder=2)
# plot circles
for i, clr in enumerate(z_eq_color_space):
   #plt.plot(z_eq_space[i]/lam, 0.0, clr+'o', ms=8, alpha=.7)
   Fz_plt.scatter(z_eq_space[i]/lam, 0.0, s=80, facecolors=clr, edgecolors='black', alpha=.7, zorder=3)
Fz_plt.legend()
Fz_plt.set_xlabel(r'Distance to the first particle, $\Delta z/\lambda$')
Fz_plt.set_ylabel(r'Optical force, $F_z$, pN')
#Fz_plt.grid(zorder=1)

# Fphi plot
for i in np.arange(len(z_eq_space)):
   Fphi_plt.plot([val*radians for val in (phi_space - np.pi)], Fphi[i]*1e12, z_eq_color_space[i], label='$F_{\phi}(\phi, \Delta z = %.2f \lambda)$' % (z_eq_space[i]/lam), xunits=radians)
Fphi_plt.set_xlabel(r'Relative twisting, $\phi$, rad')
Fphi_plt.set_ylabel(r'Optical force, $F_{\phi}$, pN')
Fphi_plt.legend()
#Fphi_plt.grid(zorder=1)

if regime == 'mm':
    plt.savefig("results/Fz_Fphi_mm_lam530_a120.pdf", bbox_inches='tight')
elif regime == 'sm':
    plt.savefig("results/Fz_Fphi_sm_lam530_a120.pdf", bbox_inches='tight')

plt.show()


# %%
### Plot Frho
#plt.figure(figsize=(8,3))
#plt.rcParams.update({'font.size': 16})
#plt.plot((rho_space - R_particle - fiber_radius)/R_particle, Frho * 1e12, 'k-', label='SM')
#plt.plot((rho_space - R_particle - fiber_radius)/R_particle, Frho2 * 1e12, 'k--', label='MM')
#plt.xlabel('$D/R_p$')
#plt.ylabel('$F^{el}_r$, pN')
##plt.grid()
#plt.legend()
#plt.show()

# %%

# Calculating data for plotting the potential well

#z_eq = 3.44 * 2*np.pi
#phi_eq = 0.0
#
#z_space = np.linspace(1.5*2*np.pi, 6.0*2*np.pi, 60)
#phi_space = np.linspace(-np.pi/3, np.pi/3, 25) + np.pi
#
#Fz = np.zeros([len(z_space), len(phi_space)])
#Fphi = np.zeros([len(z_space), len(phi_space)])
#
#start_time = time.time()
#for i, z_i in enumerate(z_space):
#    print('z step = ', i)
#    for j, phi_j in enumerate(phi_space):
#        r2 = np.array([fiber_radius + R_particle, phi_j, z_i])
#        
#        Fphi[i, j] = - force_12(1, r1, r2, R_particle, eps_particle, k, eps_out, eps_in, 
#                             fiber_radius, nmin, nmax, kzimax, E0_mod, nmin_sc, nmax_sc, case)
#        Fz[i, j] = - force_12(2, r1, r2, R_particle, eps_particle, k, eps_out, eps_in, 
#                           fiber_radius, nmin, nmax, kzimax, E0_mod, nmin_sc, nmax_sc, case)
#
### F to real units [N]
#Fphi = Fphi * E0_mod_real**2 * const.epsilon00 * 2 * np.pi * lammmm*1e-9
#Fz = Fz * E0_mod_real**2 * const.epsilon00 * 2 * np.pi * lammmm*1e-9
#
#ex_time = time.time() - start_time
#print('Exucution time: %.1f h' % (ex_time/60.0/60.0))
#
## saving results
#np.save('npy_data/F_well_data_mm', (z_space, phi_space, Fz, Fphi))


# %%
# PLOTTING

#z_space, Fz_TE = np.load('npy_data/Fz_TE_sm.npy')
#z_space, Fz_TM = np.load('npy_data/Fz_TM_sm.npy')
#z_space, Fz_TE_mm = np.load('npy_data/Fz_TE_mm.npy')
#z_space, Fz_TM_mm = np.load('npy_data/Fz_TM_mm.npy')

#z_space, Fz = np.load('npy_data/Fz.npy')
#phi_space, Fphi = np.load('npy_data/Fphi.npy')


#fig, ax = plt.subplots(figsize=[10,6])
##plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.rcParams.update({'font.size': 12})
#plt.title('$V = $ %.3f, $\lambda=$%.0f nm, $R_c = $%.0f nm, $R_p = $%.0f nm, $P_{laser}=$%.1f mW\n$\epsilon_{out}= $%.2f, $\epsilon_{fiber}= $%.2f, $\epsilon_{particle}= $%.2f,' % (V, lammmm, fiber_radius/2/np.pi*lammmm, R_particle/2/np.pi*lammmm, P_laser*1e3, eps_out, eps_in, eps_particle))
#plt.plot(z_space/lam, Fz_TE * 1e12, label='TE sm', color='b', alpha=0.7)
#plt.plot(z_space/lam, Fz_TM * 1e12, label='TM sm', color='r', alpha=0.7)
#plt.plot(z_space/lam, Fz_TE_mm * 1e12, '--',label='TE mm', color='b')
#plt.plot(z_space/lam, Fz_TM_mm * 1e12, '--',label='TM mm', color='r')

#plt.plot(z_space/lam, Fz * 1e12, label='TE', color='b', alpha=1.0)
#plt.xlabel('Distance between particles $\Delta z / \lambda$')

#plt.plot(phi_space, -Fphi * 1e12, label='TE', color='b', alpha=1.0)
#plt.xlabel(r'Relative twisting $\Delta \phi$, rad')
#
#plt.legend(shadow=True)
#plt.ylabel(r'Longitudinal force $F_z$, pN')
#
#x_label = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$",   r"$+\frac{\pi}{2}$"]
#x_tick = np.arange(-np.pi/2, np.pi/2.0 *(1 + 1/4), np.pi/4)
#plt.xticks(x_tick, x_label, fontsize=14)
##plt.xticklabels(x_label, fontsize=20)
#
#plt.grid()

#axins = zoomed_inset_axes(ax, 7, loc=4)
#axins.plot(z_space/lam, Fz_TE * 1e12, label='TE', color='b', alpha=0.7)
#axins.plot(z_space/lam, Fz_TM * 1e12, label='TM', color='r', alpha=0.7)
#plt.plot(z_space/lam, Fz_TE_mm * 1e12, '--',label='TE mm', color='b')
#plt.plot(z_space/lam, Fz_TM_mm * 1e12, '--',label='TM mm', color='r')
#plt.xticks(visible=False)
#plt.ylabel('$F_z$, pN')
#
#x1, x2, y1, y2 = 10, 11, -1.25, 1.24
#
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
#mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
#plt.grid()

#plt.show()









