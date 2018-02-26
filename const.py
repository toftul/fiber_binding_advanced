#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 01:27:22 2017

@author: ivan
"""

import numpy as np
from scipy.interpolate import interp1d


# speed of light
c = 299792458  # [m/s]
# vacuum permeability
mu0 = 4 * np.pi * 1e-7  # [H / m]


# vacuum permittivity
# final values does not depend on epsilon0
# in order to increase accuracy I put k = 1 and epsilon0 = 1
# but in final expresion for force I need to multiply by real value of epsilon0
# [F] = [p] [nabla] [E] = ... = [epsilon0] [V^2]
epsilon0 = 1.0  # [F/m]
#epsilon0 = 1.0 / (mu0 * c**2)  # [F/m]
epsilon00 = 1.0 / (mu0 * c**2)  # [F/m]

Z0 = np.sqrt(mu0 / epsilon0)

# elementary charge
e = 1.6e-19  # [C]
# electron mass
me = 9.1e-31  # [kg]

# Boltzmann constant
k_B = 1.380648528e-23  # [J/K]


# permittivities (from Optical Constants of the Noble Metals by P. B. Johnson and R. W. Christy)
# data from table in the article

energy_space = np.array([0.64, 0.77, 0.89, 1.02, 1.14, 1.26, 1.39,
                         1.51, 1.64, 1.76, 1.88, 2.01, 2.13, 2.26,
                         2.38, 2.50, 2.63, 2.75, 2.88, 3.00, 3.12,
                         3.25, 3.37, 3.50, 3.62, 3.74, 3.87, 3.99,
                         4.12, 4.24, 4.36, 4.49, 4.61, 4.74, 4.86,
                         4.98, 5.11, 5.23, 5.36, 5.48, 5.60, 5.73,
                         5.85, 5.98, 6.10, 6.22, 6.35, 6.47, 6.60])

copper_n = np.array([1.09, 0.76, 0.6, 0.48, 0.36, 0.32, 0.3, 0.26, 0.24,
                     0.21, 0.22, 0.3, 0.7, 1.02, 1.18, 1.22, 1.25, 1.24,
                     1.25, 1.28, 1.32, 1.33, 1.36, 1.37, 1.36, 1.34, 1.38,
                     1.38, 1.4, 1.42, 1.45, 1.46, 1.45, 1.41, 1.41, 1.37,
                     1.34, 1.28, 1.23, 1.18, 1.13, 1.08, 0.04, 1.01, 0.99,
                     0.98, 0.97, 0.95, 0.94])

copper_k = np.array([13.43, 11.12, 9.439, 8.245, 7.217, 6.421, 5.768,
                     5.18, 4.665, 4.205, 3.747, 3.205, 2.704, 2.577,
                     2.608, 2.564, 2.483, 2.897, 2.305, 2.207, 2.116,
                     2.045, 1.975, 1.916, 1.864, 1.821, 1.783, 1.729,
                     1.679, 1.633, 1.633, 1.646, 1.668, 1.691, 1.741,
                     1.783, 1.799, 1.802, 1.792, 1.768, 1.737, 1.699,
                     1.651, 1.599, 1.55, 1.493, 1.44, 1.388, 1.337])

silver_n = np.array([0.24, 0.15, 0.13, 0.09, 0.04, 0.04, 0.04, 0.04, 0.03,
                     0.04, 0.05, 0.06, 0.05, 0.06, 0.05, 0.05, 0.05, 0.04,
                     0.04, 0.05, 0.05, 0.05, 0.07, 0.1, 0.14, 0.17, 0.81,
                     1.13, 1.34, 1.89, 1.41, 1.41, 1.38, 1.35, 1.38, 1.31,
                     1.3, 1.28, 1.28, 1.26, 1.25, 1.22, 1.2, 1.18, 1.15,
                     1.14, 1.12, 1.1, 1.07])

silver_k = np.array([14.08, 11.85, 10.1, 8.828, 7.795, 6.992, 6.312,
                     5.727, 5.242, 4.838, 4.483, 4.152, 3.858, 3.586,
                     3.324, 3.093, 2.869, 2.657, 2.462, 2.275, 2.07,
                     1.864, 1.657, 1.419, 1.142, 0.829, 0.392, 0.616,
                     0.964, 1.161, 1.264, 1.331, 1.372, 1.387, 1.393,
                     1.389, 1.378, 1.367, 1.357, 1.344, 1.342, 1.336,
                     1.325, 1.312, 1.296, 1.277, 1.255, 1.232, 1.212])

gold_n = np.array([0.92, 0.56, 0.43, 0.35, 0.27, 0.22, 0.17, 0.16, 0.14,
                   0.13, 0.14, 0.21, 0.29, 0.43, 0.62, 1.04, 1.31, 1.38,
                   1.45, 1.46, 1.47, 1.46, 1.48, 1.5, 1.48, 1.48, 1.54,
                   1.53, 1.53, 1.49, 1.47, 1.43, 1.38, 1.35, 1.33, 1.33,
                   1.32, 1.32, 1.3, 1.31, 1.3, 1.3, 1.3, 1.3, 1.33,
                   1.33, 1.34, 1.32, 1.28])

gold_k = np.array([13.78, 11.21, 9.519, 8.145, 7.15, 6.35, 5.663,
                   5.083, 4.542, 4.103, 3.697, 3.272, 2.863, 2.455,
                   2.081, 1.833, 1.849, 1.914, 1.948, 1.958, 1.952,
                   1.933, 1.895, 1.866, 1.871, 1.883, 1.898, 1.893,
                   1.889, 1.878, 1.869, 1.847, 1.803, 1.749, 1.688,
                   1.631, 1.577, 1.536, 1.497, 1.46, 1.427, 1.387,
                   1.35, 1.304, 1.277, 1.251, 1.226, 1.203, 1.188])


# Structer of the Epsilon is:
#
# 'material': wave_length [nm], Eps', Eps''
#
h = 4.135667662e-15  # [eV s]

Epsilon = {
    'copper': np.array([1e9 * h * c / energy_space, copper_n**2 - copper_k**2, 2 * copper_n * copper_k]),
    'silver': np.array([1e9 * h * c / energy_space, silver_n**2 - silver_k**2, 2 * silver_n * silver_k]),
    'gold': np.array([1e9 * h * c / energy_space, gold_n**2 - gold_k**2, 2 * gold_n * gold_k])
}


# a function that returns a function for epsilon(lambda [nm])
def get_eps_function(material):
    f_real = interp1d(Epsilon[material][0], Epsilon[material][1], 'cubic')
    f_imag = interp1d(Epsilon[material][0], Epsilon[material][2], 'cubic')
    return f_real, f_imag
