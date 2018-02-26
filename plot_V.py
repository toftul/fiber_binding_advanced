#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:22:52 2018

@author: ivan
"""
import numpy as np
import matplotlib.pyplot as plt

eps_in = 3.5  # 2.09
eps_out = 1.77

Rf_space = np.linspace(500, 100, 9) * 1e-9
wl_space = np.linspace(200, 1200, 150) * 1e-9
Vcr = 2.405
Vcr2 = 2.55
Vcr3 = 3.6

for Rf in Rf_space:
    V = 2*np.pi/wl_space * Rf * np.sqrt(eps_in - eps_out)
    plt.plot(wl_space * 1e9, V, label="Rf = %.0f nm"% (Rf*1e9))
    
plt.plot(wl_space * 1e9, Vcr * np.ones(len(wl_space)), 'k--')
plt.plot(wl_space * 1e9, Vcr2 * np.ones(len(wl_space)), 'k--', color='gray')
plt.plot(wl_space * 1e9, Vcr3 * np.ones(len(wl_space)), 'k--', color='gray')
plt.legend()
plt.ylim(ymax=5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r'$\varepsilon_f$ = %.2f, $\varepsilon_m$ = %.2f' % (eps_in, eps_out), loc='right')
plt.xlabel(r'$\lambda$, nm')
plt.ylabel(r'$V$')
plt.show()