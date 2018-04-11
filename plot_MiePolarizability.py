"""
@author: ivan

Last change: 08.11.2017
"""

import numpy as np    
import matplotlib.pyplot as plt
import matplotlib
import lib.MiePolarizability as MiePol

matplotlib.rcParams.update({'font.size': 14})
# plt.style.use('ggplot')
lam_space = np.linspace(200, 1200, 400)
a_nm = 120
k = 2 * np.pi / lam_space * a_nm
a = 1
eps_p = 2.5
eps_m = 1.77

alpa_mie = MiePol.polarizability(k, a, eps_p, eps_m) / (4*np.pi)
alpa_dipole = MiePol.polarizability_dipole(a, eps_p, eps_m) / (4*np.pi)
alpa_dipole_rad_corr = MiePol.polarizability_dipole_rad_corr(k, a, eps_p, eps_m) / (4*np.pi)

plt.figure(figsize=(7,6))
plt.title('$R_{p} = $%.1f nm, $\epsilon_p = $%.2f, $\epsilon_m = $%.2f, ' % (a_nm, eps_p, eps_m))
plt.plot(lam_space, alpa_mie.real, 'C0', label='Mie theory (exact)')
plt.plot(lam_space, alpa_mie.imag, 'C0--')
plt.plot(lam_space, alpa_dipole_rad_corr.real, 'C1', label='Radiation correction')
plt.plot(lam_space, alpa_dipole_rad_corr.imag, 'C1--')
plt.plot(lam_space, alpa_dipole.real * k/k, 'C3', label='Dipole model')
plt.plot(lam_space, alpa_dipole.imag * k/k, 'C3--')
plt.plot(np.nan, np.nan, 'k', label='Re')
plt.plot(np.nan, np.nan, 'k--', label='Im')
plt.xlabel(r'wavelength $\lambda$, nm')
plt.ylabel(r'Polarizability $\alpha / 4\pi\epsilon_0 R_p^3$')
plt.xlim(np.min(lam_space), np.max(lam_space))
plt.legend(fontsize=12)
plt.grid()
#plt.savefig('results/polarizabllity.pdf')
plt.show()

