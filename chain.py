#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
Created on Fri May 18 12:12:16 2018 
 
@author: ivan 
""" 
 
# FUNCTIONS AND LIBS 
 
import numpy as np 
import lib.const as const 
#import GF_fiber_cython as gff 
import lib.GFfiber as gff 
import lib.GFvacuum as gfv 
import lib.MieScatCyl as MieScatCyl 
import lib.MiePolarizability as mieAlpha 
import matplotlib.pyplot as plt 
from matplotlib import gridspec 
from basic_units import radians 
from joblib import Parallel, delayed 
import time 


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
    E0 = MieScatCyl.Es(r[0], r[1], r[2], k, fiber_radius, 
                         m, E0_mod, nmin_sc, nmax_sc, case) 
 
    r_car = pol2cart(r) 
    # Incidence is the following: 
    # 
    # 
    #      /-\     k 
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
    
    
def matrixSolver(positions, R_particle, eps_particle, 
                 k, eps_out, eps_in, fiber_radius, 
                 nmin, nmax, kzimax, 
                 E0_mod, nmin_sc, nmax_sc, case): 
    """Calculates dipole moment for N particles 
    It is assumed that rho_i = rho_j 
 
    Parameters 
    ---------- 
        positions : numpy vectors 
            positions of particles in polar coordinates; 
            r1 = (rho, theta, z) 
            r2 = (rho, theta, z) 
            ... 
            rN = (rho, theta, z) 
        R_particle : float 
            particles' radius, assuming R1 = R2 
        eps_particle : complex 
            particle's epsilon (depends on k) 
     
    Returns 
    ------- 
        pColumn : numpy matrix of dipole moments 
    """ 
    
    numParticle = len(positions) 
    alpha0 = mieAlpha.polarizability(k, R_particle, eps_particle, eps_out) 
    k2_eps0 = k**2 / const.epsilon0 
    # constracting the matrix 
    I = np.eye(3) 
    for i in range(numParticle): 
        for j in range(numParticle): 
            ri, rj = positions[i], positions[j] 
            Gsij = k2_eps0 * gff.GF_pol(k, eps_out, eps_in, fiber_radius, 
                                        ri, rj, nmin, nmax, kzimax) 
            G0ij = k2_eps0 * gfv.GF_vac_pol(ri, rj, k, eps_out) 
            Gij = G0ij + Gsij 
            if j == 0:  # init the RAW 
                if i == j: 
                    RAW = I - Gsij 
                else: 
                    RAW = - Gij 
            else: 
                if i == j: 
                    RAW = np.hstack((RAW, I - Gsij)) 
                else: 
                    RAW = np.hstack((RAW, - Gsij)) 
        if i == 0:  # init the MATRIX 
            MATRIX = RAW 
        else: 
            MATRIX = np.vstack((MATRIX, RAW)) 
     
    # create b array 
    # MATRIX * pColumn = b 
    b = E0_sum(positions[0], k, fiber_radius, eps_out, eps_in, 
               E0_mod, nmin_sc, nmax_sc, case) 
    for i in np.arange(1, numParticle): 
        b = np.hstack((b, E0_sum(positions[i], k, fiber_radius, eps_out, 
                                 eps_in, E0_mod, nmin_sc, nmax_sc, case))) 
    b = alpha0 * b 
         
    # solving the matrix 
     
    pColumn = np.linalg.solve(MATRIX, b) 
     
    return(pColumn) 
    
    
def total_loc_efield(r, positions, k, case, nmin, nmax, kzimax, 
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
        r : numpy array  
            considered point 
        positions : numpy array 
            r1 = (rho, theta, z) 
            ... 
            rN = (rho, theta, z) 
        E0 : float 
            the magnitude of the incident wave 
        case : int 
            case = 1, 2 
    
    Returns 
    ------- 
        E : compex numpy array 
        E = (Erho, Etheta, Ez) 
    """ 
 
    k2_eps0 = k**2 / const.epsilon0 
     
    pColumn = matrixSolver(positions, R_particle, eps_particle, 
                           k, eps_out, eps_in, fiber_radius, 
                           nmin, nmax, kzimax, 
                           E0_mod, nmin_sc, nmax_sc, case) 
     
    E = E0_sum(r, k, fiber_radius, eps_out, eps_in, 
               E0_mod, nmin_sc, nmax_sc, case) 
    
    
    for i in range(len(positions)): 
        ri = positions[i] 
        pi = np.array([pColumn[3 * i],  
                       pColumn[3 * i + 1],  
                       pColumn[3 * i + 2]]) 
        G0 = gfv.GF_vac_pol(r, ri, k, eps_out) 
        Gs = gff.GF_pol(k, eps_out, eps_in, fiber_radius, 
                        r, ri, nmin, nmax, kzimax) 
        G = G0 + Gs 
        E += k2_eps0 * np.dot(G, pi) 
         
     
    return(E) 
     
     
     
# %% 
 
