import numpy as np 
import GF_fiber


k = 1
kzimax = 10.
eps_in = 3.
eps_out = 1.
nmax = 40
nmin = -40
r1_vec = np.array([0.55, 0, 0])
r2_vec = np.array([0.55, 0, 1])
rc = 0.5

G = GF_fiber.GF_pol_ij(k, eps_out, eps_in, rc, r1_vec, r2_vec, nmin, nmax, 0, 0, kzimax)

print(G)
