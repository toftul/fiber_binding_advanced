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
tol = 1e-8


G = GF_fiber.GF_fiber(k, eps_out, eps_in, rc, r1_vec, r2_vec, nmin, nmax, 1, 1, tol, kzimax, 0)

print(G)