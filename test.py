import numpy as np 

def f(x):
    return(np.exp(-x))

a = 3
b = 2000
nmax = 200
    
I = np.exp(-a) - np.exp(-b)

x = np.linspace(a, b, nmax)
dx = x[1] - x[0]

I1 = dx * np.sum(f(x + dx/2))
print('lin = %.2e' % (np.abs(I1 - I) / I))

x = np.geomspace(a, b, nmax)
xr = np.roll(x, 1)
dx = x - xr 
dx = np.roll(dx, -1)
dx = np.delete(dx, len(dx) - 1)
x = np.delete(x, len(x) - 1)

I2 = np.dot(dx, f(x + dx/2))
print('log = %.2e' % (np.abs(I2 - I) / I))
