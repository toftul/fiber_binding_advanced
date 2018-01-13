# -*- coding: utf-8 -*-

import numpy as np
import const
import GF_fiber as gff
import GF_vacuum as gfv
import Mie_scat_cyl
import Mie_polarizability as mie_alpha
import matplotlib.pyplot as plt


k = 1.0
lam = 2 * np.pi / k
kzimax = 15 * k
eps_out = 1.77
eps_in = 2.07
rc = lam * 0.5
nmax = 3
nmin = - nmax

r1_vec_pol = np.array([2 * rc, 0, 0])
r2_vec_pol = np.array([2 * rc, 0, 4 * rc])


gff.GF_pol(k, eps_out, eps_in, rc, r1_vec_pol, r2_vec_pol, nmin, nmax, kzimax, 
           tol=1e-8, direction=0)