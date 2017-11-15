import numpy as np 
import GF_fiber


k = 1
kzimax = 10.
eps_in = 3.
eps_out = 1.
nmax = 10
nmin = -10
r1_vec = np.array([0.55, 1, 0]) 
r2_vec = np.array([-0.55, 0, 1])
rc = 0.5

G = np.zeros([3, 3], dtype=complex)
for i in range(3):
    for j in range(3):
        G[i, j] = GF_fiber.GF_pol_ij(k, eps_out, eps_in, rc, r1_vec, r2_vec, nmin, nmax, i, j, kzimax)[0]

#print(G2.real)
#print(G2.imag)


