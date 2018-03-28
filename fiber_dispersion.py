#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:42:37 2018

@author: ivan
"""

import numpy as np
# brentq uses the classic Brent’s method to find a zero 
# of the function f on the sign changing interval [a , b]. 
# Generally considered the best of the rootfinding routines.
from scipy.optimize import brentq as root
import scipy.special as sp

# permittivities 
epsf = 3.5
epsm = 1.77
# fiber radius
rf = 450  # nm
# wavelength wl = wl0 / sqrt(eps_m)
wl = 530  # nm
k = 2*np.pi / wl

# from Wladyslaw Zakowicz, Maciej Janowicz, 
# Spontaneous emission in the presence of a dielectric cylinder
# eq (12)
def dispersion():
   qd = 1.
   qw = 1.
   rhoa = 1.
   rho = 1.
   return(0)
   
