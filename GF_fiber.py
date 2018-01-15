"""
@author: ivan

Last change: 08.11.2017
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import quad


def delta(i, j):
    if i == j:
        return(1)
    else:
        return(0)


def cart2pol(x, y):
    """
    Returns
    -------
        rho, phi : float
    """
    rho = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    return(rho, phi)


def iGNSFF11(x, k, eps_out, eps_in, rc, n, rr, rs, pr, ps, zr, zs, i, j):
    """Fiber Green's function integrand
        Calculates the integrand of the scattered part of the Green's tensor of the fiber;

    Parameters
    ----------
        x : complex
            k_z, integration variable
        k : float
            k-vector value, 2pi/\lambda_0 = \omega/c;
        eps_out, eps_in : complex
            electric permetivity outside and inside the fiber
        rc : float
            fiber radius;
        n : int
            mode order;
        rr, pr, zr : float
            reciever coordinates;
        rs, ps, zs : float
            source coordinates
        i, j : int
            rho, phi, z tensor indeces

    Returns
    -------
        iG : complex
            one component of inetegrand iGij

    """

    k1 = np.sqrt(eps_out) * k
    k1_2 = k1 * k1
    k2 = np.sqrt(eps_in) * k
    a = np.sqrt(eps_out * k*k - x*x + 0j)
    b = np.sqrt(eps_in * k*k - x*x + 0j)

    arr = a * rr
    ars = a * rs
    brc = b * rc
    arc = a * rc

    Hn1r = sp.hankel1(n, arr)
    Hn1s = sp.hankel1(n, ars)

    DHn1r = sp.h1vp(n, arr)
    DHn1s = sp.h1vp(n, ars)

    DJnb = sp.jvp(n, brc)
    Jnb = sp.jn(n, brc)

    DJna = sp.jvp(n, arc)
    Jna = sp.jn(n, arc)

    DHna = sp.h1vp(n, arc)
    Hna = sp.hankel1(n, arc)

    a2_b2 = (1 / (a*a) - 1 / (b*b))
    b2_a2_2 = a2_b2*a2_b2

    Det = rc*rc * (k2*k2 * DJnb / (b * Jnb) - k1_2 * DHna / (a * Hna)) * \
          (DJnb / (b * Jnb) - DHna / (a * Hna)) - \
          (n*n) * (x*x) * b2_a2_2
    # y = ( ( (k2^2*DJn(n,b*rc))./(b.*besselj(n,b*rc)) - (k1^2*DHn(n,a*rc))./...
    #     (a.*besselh(n,a*rc)) ).*( (DJn(n,b*rc))./(b.*besselj(n,b*rc)) - (DHn(n,a*rc))./...
    #     (a.*besselh(n,a*rc)) ) )*rc^2 - n^2*(k1^2-a.^2).*(( b.^(-2) - a.^(-2) ).^2);

    ### FRESNEL COEFFICIENTS 
    Rn11mm = - Jna / (Hna * Det) * ((k2*k2 * DJnb / (b * Jnb) - k1_2 * DHna / (a * Hna)) * \
             (DJnb / (b * Jnb) - DJna / (a * Jna)) * rc*rc - n*n * x*x * b2_a2_2)

    Rn11mn = k1 * n * rc * x * Jna / (a * Hna * Det) * a2_b2 * (DJna / Jna - DHna / Hna)
    Rn11nm = Rn11mn
    Rn11nn = Jna / Hna * (b2_a2_2 * n*n * x*x -
             (DJnb / (Jnb * b) - DHna / (Hna * a)) *
              (DJnb * k2*k2 / (Jnb * b) - DJna * k1_2 / (Jna * a)) * rc*rc) / \
             ( -b2_a2_2 * n*n * x*x + (DJnb / (Jnb * b) - 
             DHna / (Hna * a)) * (DJnb * k2*k2 / (Jnb * b) - DHna * k1_2 / (Hna * a)) * rc*rc)

    iG = 0j
    # rr component
    if i == 0 and j == 0:
        iGNrr11mm = Hn1r * Hn1s * n*n * Rn11mm / (rr * rs * a*a)
        iGNrr11nm = DHn1r * Hn1s * n * Rn11nm * x / (k1 * rs * a)
        iGNrr11mn = DHn1s * Hn1r * n * Rn11mn * x / (k1 * rr * a)
        iGNrr11nn = DHn1r * DHn1s * Rn11nn * x*x / k1_2
          
        iG = (2 - delta(n, 0)) * 1j * np.cos(n * (pr - ps)) * (iGNrr11mm +
             iGNrr11nm + iGNrr11mn + iGNrr11nn) * np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # rp component
    elif i == 0 and j == 1:
        iGNrp11mm = n * Hn1r * DHn1s * Rn11mm / (rr * a)
        iGNrp11nm = x * DHn1r * DHn1s * Rn11nm / k1
        iGNrp11mn = Hn1r * Hn1s * n*n * x * Rn11mn / (k1 * rr * rs * a*a)
        iGNrp11nn = DHn1r * Hn1s * n * Rn11nn * x*x / (k1_2 * rs * a)

        iG = (2 - delta(n, 0)) * 1j * (iGNrp11mm + iGNrp11nm +
             iGNrp11mn + iGNrp11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # rz component
    elif i == 0 and j == 2:
        iGNrz11mn = 1j * Hn1r * Hn1s * n * Rn11mn / (k1 * rr)
        iGNrz11nn = 1j * a * DHn1r * Hn1s * Rn11nn * x / (k1_2)

        iG = (2 - delta(n, 0)) * 1j * (
             iGNrz11mn + iGNrz11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # pr component
    elif i == 1 and j == 0:
        iGNpr11mm = - DHn1r * Hn1s * n * Rn11mm / (rs * a)
        iGNpr11nm = - Hn1r * Hn1s * n*n * Rn11nm * x / (k1 * rr * rs * a*a)
        iGNpr11mn = - DHn1r * DHn1s * Rn11mn * x / k1
        iGNpr11nn = - DHn1s * Hn1r * n * Rn11nn * x*x / (k1_2 * rr * a)

        iG = (2 - delta(n, 0)) * 1j * (iGNpr11mm + iGNpr11nm +
             iGNpr11mn + iGNpr11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)    
    # pp component
    elif i == 1 and j == 1:
        iGNpp11mm = Rn11mm * DHn1r * DHn1s
        iGNpp11nm = n * x * Rn11nm * Hn1r * DHn1s / (k1 * rr * a)
        iGNpp11mn = n * Rn11mn * DHn1r * Hn1s * x / (k1 * rs * a)
        iGNpp11nn = n*n * x*x * Rn11nn * Hn1r * Hn1s / (k1_2 * rr * rs * a*a)

        iG = (2 - delta(n, 0)) * 1j * (iGNpp11mm + iGNpp11nm +
             iGNpp11mn + iGNpp11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # pz component
    elif i == 1 and j == 2:
        iGNpz11mn = - 1j * a * DHn1r * Hn1s * Rn11mn / k1
        iGNpz11nn = - 1j * Hn1r * Hn1s * n * Rn11nn * x / (k1_2 * rr)

        iG = (2 - delta(n, 0)) * 1j * (
             iGNpz11mn + iGNpz11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zr component
    elif i == 2 and j == 0:
        iGNzr11nm = - 1j * Hn1r * Hn1s * n * Rn11nm / (k1 * rs)
        iGNzr11nn = - 1j * a * DHn1s * Hn1r * Rn11nn * x / k1_2

        iG = (2 - delta(n, 0)) * 1j * (iGNzr11nm + 
             iGNzr11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zp component
    elif i == 2 and j == 1:
        iGNzp11nm = -1j * a * DHn1s * Hn1r * Rn11nm / k1
        iGNzp11nn = -1j * Hn1r * Hn1s * n * Rn11nn * x / (k1_2 * rs)

        iG = (2 - delta(n, 0)) * 1j * (iGNzp11nm +
            iGNzp11nn) * np.sin(n * (pr - ps)) * \
            np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zz component
    elif i == 2 and j == 2:
        iG = (2 - delta(n, 0)) * 1j * Rn11nn * Hn1r * \
             Hn1s * a*a * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (k1_2 * 8 * np.pi)

    return iG


def KOSTYL(t, area, im_max, re_max, theta,
           k, eps_out, eps_in, rc, n, rr, rs, pr, ps, zr, zs, i, j):
    # contour parametrization
    z = 0.0

    """
    AREAS:

                   | Im kz
             *     |
         (2) * *   |(3)
             *   * |            (5)       Re kz
    **********-----*-----**************-----
      (1)          | *   *
                   |   * * (4)
                   |     *
                   |
    """

    if area == 3:
        z = t * np.exp(1j * theta)
    elif area == 2:
        z = 1j * t - re_max
    elif area == 1:
        z = t
    elif area == 4:
        z = 1j * t + re_max
    elif area == 5:
        z = t

    return(iGNSFF11(z, k, eps_out, eps_in, rc, n, rr, rs, pr, ps, zr, zs, i, j))


def GF_pol_ij(k, eps_out, eps_in, rc, r1_vec, r2_vec, nmin, nmax,
             i, j, kzimax, tol=1e-8):
    """Fiber Green's function
    
    Symmetry properties
    -------------------
        
                | G11 G12 G13 |
    G(r1, r2) = | G21 G22 G23 |
                | G31 G32 G33 |
                     
                | G11 ... G31 |
    G(r2, r1) = | ... ... G32 |
                | G13 G23 G33 |
    
    where '...' means something different

    Parameters
    ----------
        k : float
            k-vector value, 2pi/\lambda_0 = \omega/c;
        eps_out, eps_in : complex
            electric permetivity outside and inside the fiber
        rc : float 
            fiber radius;
        r1_vec, r2_vec : numpy 2D array
            positions of the source and the reciever,
            r = (rho, theta, z)
        nmin : int
            minimal mode number in the sum;
        nmac : int
            maximal mode number in the sun;
        i, j : int 
            tensor indicies over cylindrical coordinates \rho, \phi, z;
        tol : int 
            relative tolerance for the sum (how many modes 'n' to consider?); 
            |G^{N} - G^{N-1}|/G^{N}; 
        kzimax : float
            upper integration limit in kz,
            usually several k_in;

    Returns
    -------
        G : complex
            one component of Gij
        nmodes : number of considered modes
            condition for cutting the 'n' exists

    """
#    def complex_quad(func, a, b, limit=50, *args):
#        def RE(x):
#            return func(x, *args).real
#    
#        def IM(x):
#            return func(x, *args).imag
#    
#        return(quad(RE, a, b, limit=limit)[0] + 
#               1j * quad(IM, a, b, limit=limit)[0])

    # cartesian to polar (projectures)
    r1, p1, z1 = r1_vec[0], r1_vec[1], r1_vec[2]
    r2, p2, z2 = r2_vec[0], r2_vec[1], r2_vec[2]

    k2 = np.sqrt(eps_in) * k

    kzReMax = kzimax * np.sqrt(eps_in)
    kzImMax = k * 1e-2  # choose this quantity to be smaller for 
                        # larger interatomic distances dz, since exp(1i k_z delta_z) 
    rel = 0.0

    """
    AREAS:

                   | Im kz
             *     |
         (2) * *   |(3)
             *   * |            (5)       Re kz
    **********-----*-----**************-----
      (1)          | *   *
                   |   * * (4)
                   |     *
                   |
    """

    GNS11mat = 0.  # Green's tensor component
    Gnprev = 1e9  # previous, it sums from nmin to nmax-1; when nmax = 0 Gnprev = 0;

    theta = - np.arctan(kzImMax / (1.1 * k2))
    
    MaxIntervalCount = 150
    # k numpy arrays in log scale for integration
    ks5 = np.geomspace(1.1 * k2, kzReMax, MaxIntervalCount*0.2)
    ks5r = np.roll(ks5, 1)
    dks5 = ks5 - ks5r
    dks5 = np.roll(dks5, -1)
    dks5 = np.delete(dks5, len(dks5) - 1)
    ks5 = np.delete(ks5, len(ks5) - 1)
    
    ks2 = np.geomspace(0.0001 * kzImMax, kzImMax, MaxIntervalCount*0.2)
    ks2r = np.roll(ks2, 1)
    dks2 = ks2 - ks2r
    dks2 = np.roll(dks2, -1)
    dks2 = np.delete(dks2, len(dks2) - 1)
    ks2 = np.delete(ks2, len(ks2) - 1)
    
#    ks32 = np.geomspace(1e-6 * k, np.sqrt(kzImMax*kzImMax + (1.1 * k2)*(1.1 * k2)),
#                        2* MaxIntervalCount)
#    ks32r = np.roll(ks32, 1)
#    dks32 = ks32 - ks32r
#    dks32 = np.roll(dks32, -1)
#    dks32 = np.delete(dks32, len(dks32) - 1)
#    ks32 = np.delete(ks32, len(ks32) - 1)
#    
#    ks31 = - ks32
#    dks31 = dks32
    
    ks3 = np.linspace(- np.sqrt(kzImMax*kzImMax + (1.1 * k2)*(1.1 * k2)), 
                      np.sqrt(kzImMax*kzImMax + (1.1 * k2)*(1.1 * k2)), MaxIntervalCount)
    dks3 = ks3[1] - ks3[0]
    
    ks4 = - ks2
    dks4 = dks2
    
    ks1 = -ks5
    dks1 = dks5
    
    
    # REGULAR CASE: ANY STRUCTURE, ALL MODES
    for num in range(nmin, nmax + 1):
        # direct integration using numpy array calculation
        # area = 1
        y1 = iGNSFF11(ks1 - dks1 * 0.5, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
        # area = 2
        y2 = iGNSFF11((ks2 - dks2 * 0.5)*1j - 1.1 * k2, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
        # area = 3
        eexxpp = np.exp(1j * theta)
        y3 = iGNSFF11((ks3 - dks3)*eexxpp, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j) + \
             4 * iGNSFF11((ks3 - 0.5 * dks3)*eexxpp, 
                          k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j) + \
             iGNSFF11(ks3*eexxpp, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
        # area = 4
        y4 = iGNSFF11((ks4 + dks4 * 0.5)*1j + 1.1 * k2, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
        # area = 5
        y5 = iGNSFF11(ks5 + dks5 * 0.5, 
                      k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
  
        GNS11mat += np.dot(dks1, y1) + 1j * np.dot(dks2, y2) + \
                    1j * np.dot(dks4, y4) + np.dot(dks5, y5) + \
                    dks3/6 * np.sum(y3) * np.exp(1j * theta)
        
#        # area = 1
#        GNS11mat += complex_quad(KOSTYL, -kzReMax, -1.1 * k2, MaxIntervalCount,
#                                 1, kzImMax, 1.1 * k2, theta, k,
#                                 eps_out, eps_in, rc, num, r1, r2,
#                                 p1, p2, z1, z2, i, j)
#        # area = 2
#        GNS11mat += 1j * complex_quad(KOSTYL, 0.0, kzImMax, MaxIntervalCount,
#                                 2, kzImMax, 1.1 * k2, theta, k,
#                                 eps_out, eps_in, rc, num, r1, r2,
#                                 p1, p2, z1, z2, i, j)
#        # area = 3
#        GNS11mat += np.exp(1j * theta) \
#                * complex_quad(KOSTYL, - np.sqrt(kzImMax*kzImMax + (1.1 * k2)*(1.1 * k2)),
#                                         np.sqrt(kzImMax*kzImMax + (1.1 * k2)*(1.1 * k2)),
#                                 MaxIntervalCount,
#                                 3, kzImMax, 1.1 * k2, theta, k,
#                                 eps_out, eps_in, rc, num, r1, r2,
#                                 p1, p2, z1, z2, i, j)
#        # area = 4
#        GNS11mat += 1j * complex_quad(KOSTYL, - kzImMax, 0.0, MaxIntervalCount,
#                                 4, kzImMax, 1.1 * k2, theta, k,
#                                 eps_out, eps_in, rc, num, r1, r2,
#                                 p1, p2, z1, z2, i, j)
#        # area = 5
#        GNS11mat += complex_quad(KOSTYL, 1.1 * k2, kzReMax, MaxIntervalCount,
#                                 5, kzImMax, 1.1 * k2, theta, k,
#                                 eps_out, eps_in, rc, num, r1, r2,
#                                 p1, p2, z1, z2, i, j)
        
        ## Direct integration along Real axis
        #GNS11mat += complex_quad(iGNSFF11, -kzReMax, kzReMax, MaxIntervalCount,
        #                         k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)

        # relative contribution of the current mode 'n'
        normGnprev = np.abs(Gnprev)
        if normGnprev == 0:
            rel = np.abs(normGnprev - np.abs(GNS11mat))
        else:
            rel = np.abs(normGnprev - np.abs(GNS11mat)) / normGnprev
        # condition for cutting the 'n'
        if rel < tol:
            break
        Gnprev = GNS11mat

    return GNS11mat

def GF_pol(k, eps_out, eps_in, rc, r1_vec_pol, r2_vec_pol,
           nmin, nmax, kzimax, tol=1e-8):
    '''Returns full tensor Gs in polar coordinates
    It is assumed that rho1 = rho2
    
    Parameters
    ----------
        r1_vec, r2_vec : numpy arrays
            in polar coordinates (rho, theta, z)
    '''
    
    Gs = np.zeros([3, 3], dtype=complex)
    for i in range(3):
        for j in range(i, 3):
            Gs[i, j] = GF_pol_ij(k, eps_out, eps_in,
                                     rc, r1_vec_pol, r2_vec_pol,
                                     nmin, nmax, i, j, kzimax)
            if i != j:
                if i == 0:
                    Gs[j, i] = - Gs[i, j]
                else:
                    Gs[j, i] = Gs[i, j]
          
    return(Gs)
    

# not yet done!
def GF_car(k, eps_out, eps_in, rc, r1_vec, r2_vec,
          nmin, nmax, kzimax, tol=1e-8):
    '''Returns full tensor Gs in Cartesian coordinates
    
    Parameters
    ----------
        r1_vec, r2_vec : numpy arrays
            in cart. coordinates
    '''
    Gs = np.zeros([3, 3], dtype=complex)
    
    r1p = np.zeros(3)
    r2p = r1p
    
    r1p[0], r1p[1] = cart2pol(r1_vec[0], r1_vec[1])
    r1p[2] = r1_vec[2]
    
    r2p[0], r2p[1] = cart2pol(r2_vec[0], r2_vec[1])
    r2p[2] = r2_vec[2]

    for i in range(3):
        for j in range(i, 3):
            Gs[i, j] = GF_pol_ij(k, eps_out, eps_in,
                                rc, r1p, r2p,
                                nmin, nmax, i, j, kzimax)
            if i != j:
                if i == 0:
                    Gs[j, i] = - Gs[i, j]
                else:
                    Gs[j, i] = Gs[i, j]

    theta1 = r1p[1]
    theta2 = r2p[1]
    Q1 = np.array([[np.cos(theta1), np.sin(theta1), 0],
                  [-np.sin(theta1), np.cos(theta1), 0],
                  [0, 0, 1]])
    Q2T = np.array([[np.cos(theta2), -np.sin(theta2), 0],
                  [np.sin(theta2), np.cos(theta2), 0],
                  [0, 0, 1]])

    # T_car = Q1 T_pol Q2^T
    return(Q1 @ Gs @ Q2T)