#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
Created on Sat May 12 12:06:10 2018 
 
@author: ivan 
""" 
 
import numpy as np 
from scipy.integrate import quad 
import matplotlib.pyplot as plt 
 
 
def integrand_drho(H, A, y, phi): 
    def f(rho): 
        C1 = H + A + y + rho * np.cos(phi) 
        return(rho * C1 / (C1 * C1 + rho*rho*np.sin(phi)*np.sin(phi))**3.5) 
    return(quad(f, 0, A)[0]) 
     
def integrand_dphi(H, A, y): 
    f = lambda phi: integrand_drho(H, A, y, phi) 
    return(quad(f, 0, np.pi)[0]) 
         
def FvdW(H, A, a, A123): 
    int_y = lambda y: (2 - y) * y * integrand_dphi(H, A, y) 
    f = -15/4 * quad(int_y, 0, 2)[0] 
    return(A123/a * f) 
    
# %% 
 
rp = 120e-9  # [m]  particle radius  
rfMM = 495e-9  # [m]  fiber radius 
rfSM = 130e-9  # [m]  fiber radius 
     
# A = R / a 
ASM = rfSM / rp 
AMM = rfMM / rp 
A123 = 10e-20  # [J]  approx value for the Hamaker constant 
 
 
# gap between particle and fiber surface 
gapSpace = np.logspace(np.log10(rp/50), np.log10(5*rp), 30)  # [m]   
FSpaceSM = np.zeros(len(gapSpace)) 
FSpaceMM = np.zeros(len(gapSpace)) 
for i, gap in enumerate(gapSpace): 
    print(i/len(gapSpace)) 
    FSpaceSM[i] = FvdW(gap/rp, ASM, rp, A123) 
    FSpaceMM[i] = FvdW(gap/rp, AMM, rp, A123) 
 
    
# %% 
plt.rcParams.update({'font.size': 14}) 
plt.figure(figsize=(8,5)) 
#plt.title('Attractive vdW force') 
plt.semilogy(gapSpace/rp, np.abs(FSpaceSM)*1e12, 'k-', label='SM') 
plt.semilogy(gapSpace/rp, np.abs(FSpaceMM)*1e12, 'k--', label='MM') 
plt.xlabel('Relative gap width, $g/R_p$') 
plt.ylabel('vdW attractive force $|F^{wdW}_r|$, pN') 
#plt.xlim((1e-2, 1e1)) 
#plt.xlim((np.min(gapSpace/rp), np.max(gapSpace/rp))) 
plt.ylim((0.1*np.min(np.abs(FSpaceSM)*1e12), 10*np.max(np.abs(FSpaceSM)*1e12))) 
plt.legend() 
#plt.grid(True,which="both",ls="-", alpha=0.4) 
#plt.savefig('results/vdw.pdf') 
plt.show() 


# %% 
### sanity check 
# Compare with 
# 'The van der Waals Interaction between a Spherical Particle and a Cylinder' 
 
#A = 1 
# 
#Hspace = np.logspace(-2, 2, num=10) 
#FSpace = np.zeros(len(Hspace)) 
#for i, H in enumerate(Hspace): 
#    print(i) 
#    FSpace[i] = FvdW(H, A, 1, 1) 
 
# %% 
# plot 
 
#plt.figure(figsize=(8,5)) 
#plt.rcParams.update({'font.size': 16}) 
#plt.loglog(Hspace, -FSpace, '-k') 
#plt.xlabel('D, nm') 
#plt.ylabel('$F^{wdW}_r$, pN') 
#plt.grid() 
#plt.show() 
 
# STATUS: CHACKED! 