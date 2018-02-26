# -*- coding: utf-8 -*-

import numpy as np
#import const
import GF_fiber as gff
#import GF_fiber_cython as gff
#import GF_vacuum as gfv
#import Mie_scat_cyl
#import Mie_polarizability as mie_alpha
import matplotlib.pyplot as plt
import time

pi = np.pi

k0 = 1
eps1 = 1.0
eps2 = 2.09
rc = 1 / 3 * 2 * pi /k0
# corresponds to a pic in Zakowicz
drho = 1 / 6 * 2 * pi /k0
zr = 0.0 * 2 * pi /k0
zmin = 0.1 * 2 * pi /k0
zmax = 15.0 * 2 * pi /k0
zlen = 150
zvec = np.linspace(zmin, zmax, zlen)
V = k0 * rc * np.sqrt(eps2 - eps1)
print('V = ', V)

nmin = 0
nmax = 3

p_d = 0.0 * 2 * pi

z_d = 0.0 * 2 * pi

k_in = np.sqrt(eps2) * k0
k_out = np.sqrt(eps1) * k0

kzimax = 3 * np.sqrt(eps2) * k0

g_sc = np.zeros(zlen, dtype=complex)
start_time = time.time()
for j in range(zlen):
    print('r: %.2f' % (j / zlen))
    r1_vec = np.array([rc + drho, p_d, zr])
    r2_vec = np.array([rc + drho, p_d, zvec[j]])
    i_int, j_int = 1, 1
    g_sc[j] = gff.GF_pol_ij(k0, eps1, eps2, rc, r1_vec, r2_vec,
                            nmin, nmax, i_int, j_int, kzimax)

T = time.time() - start_time
print("\Time: %.2f s" % (T))

plt.plot(zvec / (2 * pi), g_sc.imag, label=r'Im$G_{zz}$')
plt.plot(zvec / (2 * pi), g_sc.real, label=r'Re$G_{zz}$')
plt.legend()
plt.xlabel(r'$\Delta z / \lambda$, nm')
plt.ylabel(r'$G_{zz}$')
plt.show()

