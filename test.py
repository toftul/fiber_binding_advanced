# -*- coding: utf-8 -*-

import numpy as np
import GF_vacuum as gfv

k = 1
R = np.array([1, 2, 3]) * 2*np.pi / k

G1 = gfv.GF_vac(R, k)


k = 2*np.pi/550e-9
R = np.array([1, 2, 3]) * 2*np.pi / k

G2 = gfv.GF_vac(R, k)

