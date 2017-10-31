import numpy as np
import scipy.special as sp
from scipy.integrate import quad


def complex_quad(func, a, b, limit=50, *args):
    def RE(x):
        return func(x, *args).real

    def IM(x):
        return func(x, *args).imag

    return(quad(RE, a, b, limit=limit)[0] + 
           1j * quad(IM, a, b, limit=limit)[0])


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
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
    k2 = np.sqrt(eps_in) * k
    a = np.sqrt(eps_out * k**2 - x**2 + 0j)
    b = np.sqrt(eps_in * k**2 - x**2 + 0j)

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

    a2_b2 = (1 / a**2 - 1 / b**2)
    b2_a2_2 = a2_b2**2

    Det = rc**2 * (k2**2 * DJnb / (b * Jnb) - k1**2 * DHna / (a * Hna)) * \
          (DJnb / (b * Jnb) - DHna / (a * Hna)) - \
          (n**2) * (x**2) * b2_a2_2
    # y = ( ( (k2^2*DJn(n,b*rc))./(b.*besselj(n,b*rc)) - (k1^2*DHn(n,a*rc))./...
    #     (a.*besselh(n,a*rc)) ).*( (DJn(n,b*rc))./(b.*besselj(n,b*rc)) - (DHn(n,a*rc))./...
    #     (a.*besselh(n,a*rc)) ) )*rc^2 - n^2*(k1^2-a.^2).*(( b.^(-2) - a.^(-2) ).^2);

    ### FRESNEL COEFFICIENTS 
    Rn11mm = - Jna / (Hna * Det) * ((k2**2 * DJnb / (b * Jnb) - k1**2 * DHna / (a * Hna)) * \
             (DJnb / (b * Jnb) - DJna / (a * Jna)) * rc**2 - n**2 * x**2 * b2_a2_2)

    Rn11mn = k1 * n * rc * x * Jna / (a * Hna * Det) * a2_b2 * (DJna / Jna - DHna / Hna)
    Rn11nm = Rn11mn
    Rn11nn = Jna / Hna * (b2_a2_2 * n**2 * x**2 -
             (DJnb / (Jnb * b) - DHna / (Hna * a)) *
              (DJnb * k2**2 / (Jnb * b) - DJna * k1**2 / (Jna * a)) * rc**2) / \
             ( -b2_a2_2 * n**2 * x**2 + (DJnb / (Jnb * b) - 
             DHna / (Hna * a)) * (DJnb * k2**2 / (Jnb * b) - DHna * k1**2 / (Hna * a)) * rc**2)

    iG = 0j
    # rr component
    if i == 1 and j == 1:
        iGNrr11mm = Hn1r * Hn1s * n**2 * Rn11mm / (rr * rs * a**2)
        iGNrr11nm = DHn1r * Hn1s * n * Rn11nm * x / (k1 * rs * a)
        iGNrr11mn = DHn1s * Hn1r * n * Rn11mn * x / (k1 * rr * a)
        iGNrr11nn = DHn1r * DHn1s * Rn11nn * x**2 / k1**2
          
        iG = (2 - np.kron(n, 0)) * 1j * np.cos(n * (pr - ps)) * (iGNrr11mm +
             iGNrr11nm + iGNrr11mn + iGNrr11nn) * np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # rp component
    elif i == 1 and j == 2:
        iGNrp11mm = n * Hn1r * DHn1s * Rn11mm / (rr * a)
        iGNrp11nm = x * DHn1r * DHn1s * Rn11nm / k1
        iGNrp11mn = Hn1r * Hn1s * n**2 * x * Rn11mn / (k1 * rr * rs * a**2)
        iGNrp11nn = DHn1r * Hn1s * n * Rn11nn * x**2 / (k1**2 * rs * a)

        iG = (2 - np.kron(n, 0)) * 1j * (iGNrp11mm + iGNrp11nm +
             iGNrp11mn + iGNrp11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # rz component
    elif i == 1 and j == 3:
        iGNrz11mn = 1j * Hn1r * Hn1s * n * Rn11mn / k1 / rr
        iGNrz11nn = 1j * a * DHn1r * Hn1s * Rn11nn * x / k1 / k1

        iG = (2 - np.kron(n, 0)) * 1j * (
             iGNrz11mn + iGNrz11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # pr component
    elif i == 2 and j == 1:
        iGNpr11mm = - DHn1r * Hn1s * n * Rn11mm / (rs * a)
        iGNpr11nm = - Hn1r * Hn1s * n**2 * Rn11nm * x / (k1 * rr * rs * a**2)
        iGNpr11mn = - DHn1r * DHn1s * Rn11mn * x / k1
        iGNpr11nn = - DHn1s * Hn1r * n * Rn11nn * x**2 / (k1**2 * rr * a)

        iG = (2 - np.kron(n, 0)) * 1j * (iGNpr11mm + iGNpr11nm +
             iGNpr11mn + iGNpr11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)    
    # pp component
    elif i == 2 and j == 2:
        iGNpp11mm = Rn11mm * DHn1r * DHn1s
        iGNpp11nm = n * x * Rn11nm * Hn1r * DHn1s / (k1 * rr * a)
        iGNpp11mn = n * Rn11mn * DHn1r * Hn1s * x / (k1 * rs * a)
        iGNpp11nn = n**2 * x**2 * Rn11nn * Hn1r * Hn1s / (k1**2 * rr * rs * a**2)

        iG = (2 - np.kron(n, 0)) * 1j * (iGNpp11mm + iGNpp11nm +
             iGNpp11mn + iGNpp11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # pz component
    elif i == 2 and j == 3:
        iGNpz11mn = - 1j * a * DHn1r * Hn1s * Rn11mn / k1
        iGNpz11nn = - 1j * Hn1r * Hn1s * n * Rn11nn * x / (k1**2 * rr)

        iG = (2 - np.kron(n, 0)) * 1j * (
             iGNpz11mn + iGNpz11nn) * np.sin(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zr component
    elif i == 3 and j == 1:
        iGNzr11nm = - 1j * Hn1r * Hn1s * n * Rn11nm / (k1 * rs)
        iGNzr11nn = - 1j * a * DHn1s * Hn1r * Rn11nn * x / k1**2

        iG = (2 - np.kron(n, 0)) * 1j * (iGNzr11nm + 
             iGNzr11nn) * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zp component
    elif i == 3 and j == 2:
        iGNzp11nm = -1j * a * DHn1s * Hn1r * Rn11nm / k1
        iGNzp11nn = -1j * Hn1r * Hn1s * n * Rn11nn * x / (k1**2 * rs)

        iG = (2 - np.kron(n, 0)) * 1j * (iGNzp11nm +
            iGNzp11nn) * np.sin(n * (pr - ps)) * \
            np.exp(1j * x * (zr - zs)) / (8 * np.pi)
    # zz component
    elif i == 3 and j == 3:
        iG = (2 - np.kron(n, 0)) * 1j * Rn11nn * Hn1r * \
             Hn1s * a**2 * np.cos(n * (pr - ps)) * \
             np.exp(1j * x * (zr - zs)) / (k1**2 * 8 * np.pi)

    return iG


def KOSTYL(t, direction, area, im_max, re_max, theta,
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

    # forward
    if direction == 1:
        if area == 3:
            z = t * np.exp(1j * theta)
        elif area == 4:
            z = 1j * t + re_max
        elif area == 5:
            z = t
    # backward
    elif direction == -1:
        if area == 3:
            z = t * np.exp(1j * theta)
        elif area == 2:
            z = 1j * t - re_max
        elif area == 1:
            z = t

    # both
    elif direction == 0:
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


def GF_fiber(k, eps_out, eps_in, rc, r1_vec, r2_vec, nmin, nmax,
             i, j, tol, kzimax, direction):
    """Fiber Green's function

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
            r = (x, y, z)
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
        direction : int
            diraction of propagation,
            +1 -- forward,
            -1 -- backward,
            0 -- both;


    Returns
    -------
        G : complex
            one component of Gij
        nmodes : number of considered modes
            condition for cutting the 'n' exists

    """

    # cartesian to polar 
    r1, p1 = cart2pol(r1_vec[0], r1_vec[1])
    z1 = r1_vec[2]
    r2, p2 = cart2pol(r2_vec[0], r2_vec[1])
    z2 = r2_vec[2]

    k2 = np.sqrt(eps_in) * k

    kzReMax = kzimax * np.sqrt(eps_in)
    kzImMax = k * 1e-4  # choose this quantity to be smaller for 
                        # larger interatomic distances dz, since exp(1i k_z delta_z) 


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
    nmodes = 0.  # number of considered 'n' modes
    Gnprev = 0.  # previous, it sums from nmin to nmax-1; when nmax = 0 Gnprev = 0;

    theta = - np.arctan(kzImMax / (1.1 * k2))
    MaxIntervalCount = 100000
    for num in range(nmin, nmax + 1):
        # REGULAR CASE: ANY STRUCTURE, ALL MODES
        if direction == 0:
            # area = 1
            GNS11mat += complex_quad(KOSTYL, -kzReMax, -1.1 * k2, MaxIntervalCount,
                                     direction, 1, kzImMax, 1.1 * k2, theta, k,
                                     eps_out, eps_in, rc, num, r1, r2,
                                     p1, p2, z1, z2, i, j)
            # area = 2
            GNS11mat += 1j * complex_quad(KOSTYL, 0.0, kzImMax, MaxIntervalCount,
                                     direction, 1, kzImMax, 1.1 * k2, theta, k,
                                     eps_out, eps_in, rc, num, r1, r2,
                                     p1, p2, z1, z2, i, j)
            # area = 3
            GNS11mat += np.exp(1j * theta) \
                    * complex_quad(KOSTYL, - np.sqrt(kzImMax**2 + (1.1 * k2)**2),
                                             np.sqrt(kzImMax**2 + (1.1 * k2)**2),
                                     MaxIntervalCount,
                                     direction, 1, kzImMax, 1.1 * k2, theta, k,
                                     eps_out, eps_in, rc, num, r1, r2,
                                     p1, p2, z1, z2, i, j)
            # area = 4
            GNS11mat += 1j * complex_quad(KOSTYL, - kzImMax, 0.0, MaxIntervalCount,
                                     direction, 1, kzImMax, 1.1 * k2, theta, k,
                                     eps_out, eps_in, rc, num, r1, r2,
                                     p1, p2, z1, z2, i, j)
            # area = 5
            GNS11mat += complex_quad(KOSTYL, 1.1 * k2, kzReMax, MaxIntervalCount,
                                     direction, 1, kzImMax, 1.1 * k2, theta, k,
                                     eps_out, eps_in, rc, num, r1, r2,
                                     p1, p2, z1, z2, i, j)

            ## Direct integration along Real axis
            #GNS11mat += complex_quad(iGNSFF11, -kzReMax, kzReMax, MaxIntervalCount,
            #                         k, eps_out, eps_in, rc, num, r1, r2, p1, p2, z1, z2, i, j)
        else:
            GNS11mat = 0.
            print('GF_fiber ERROR: direction != -1, 0, 1')

        # relative contribution of the current mode 'n'
        rel = np.abs(np.abs(GNS11mat) - np.abs(Gnprev)) / np.abs(GNS11mat)
        # condition for cutting the 'n'
        if rel < tol:
            break
        Gnprev = GNS11mat
        nmodes += 1

    return GNS11mat, nmodes
