# -*- coding: utf-8 -*-

import numpy as np
import const
import GF_fiber_cython as gff
#import GF_fiber as gff
import GF_vacuum as gfv
import Mie_scat_cyl
import Mie_polarizability as mie_alpha
import matplotlib.pyplot as plt
import time



k = 1
lam = 2 * np.pi / k
kzimax = 4 * k
eps_out = 1.77
eps_in = 2.07
rc = lam * 0.5
nmax = 3
nmin = - nmax

r1_vec_pol = np.array([2 * rc, 0, 0])
r2_vec_pol = np.array([2 * rc, 0, 4 * rc])

# Before optimization
T0 = 224.744238615036 

start_time = time.time()
G = gff.GF_pol(k, eps_out, eps_in, rc, r1_vec_pol, r2_vec_pol, nmin, nmax, kzimax)
T = time.time() - start_time
print("\nResult: %.2f times faster" % (T0/T))