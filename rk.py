import numpy as np
import numpad as ad

class CashTable:
    '''
    Diagonally Implicit Runge Kutta (DIRK) coefficients in Section 2.1 of
    Cash. IMA J Appl Math (1979) 24 (3): 293-301. doi: 10.1093/imamat/24.3.293
    '''
    a = np.polynomial.polynomial.polyroots([-1./6, 3./2, -3, 1])[1]
    t2 = (a**2 - 3./2*a + 1./3) / (a**2 - 2*a + 1./2)
    b1 = (t2/2 - 1./6) / ((t2 - a) * (1 - a))
    b2 = (a/2 - 1./6) / ((a - t2) * (1 - t2))
    c1 = (t2 - 1./2) / (t2 - a)
    c2 = (a - 1./2) / (a - t2)

def step(f, u0, dt, args=(), argv={}):
    ff = lambda u : f(u, *args, **argv)

    step1 = lambda u : u - u0 - dt * CashTable.a * ff(u)
    u1 = ad.solve(step1, u0, verbose=False)
    f1 = ff(u1)

    step2 = lambda u : u - u0 - dt * ((CashTable.t2 - CashTable.a) * f1 \
                                    + CashTable.a * ff(u))
    u2 = ad.solve(step2, u1, verbose=False)
    f2 = ff(u2)

    step3 = lambda u : u - u0 - dt * (CashTable.b1 * f1 + CashTable.b2 * f2 \
                                    + CashTable.a * ff(u))
    u3 = ad.solve(step3, u2, verbose=False)

    u_second_order = u0 + dt * (CashTable.c1 * f1 + CashTable.c2 * f2)
    return u3, u_second_order

def roundTo2ToK(n):
    log2n = log2(max(1, n))
    return 2**int(round(log2n))

def integrate(f, u0, t, relTol=1E-4, absTol=1E-6):
    u = ad.array(u0).copy()
    uHistory = [u]
    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        nSubdiv = roundTo2ToK((t[i+1] - t[i]) / dt)
        dt = (t[i+1] - t[i]) / nSubdiv
        j = 0
        while j < nSubdiv:
            u3rd, u2nd = step(f, u, dt)
            uNorm = np.linalg.norm(ad.value(u))
            errNorm = np.linalg.norm(ad.value(u3rd) - ad.value(u2nd))
            if errNorm > max(absTol, relTol * uNorm):
                dt, j, nSubdiv = 0.5 * dt, 2 * j, 2 * nSubdiv
            else:
                j += 1
                u = u3rd
                if errNorm < 0.25 * max(absTol, relTol * uNorm) and \
                        j % 2 == 0 and nSubdiv > 1:
                    dt, j, nSubdiv = 2 * dt, j / 2, nSubdiv / 2
                    print(dt)
        assert j == nSubdiv
        uHistory.append(u)
    return ad.array(uHistory)

if __name__ == '__main__':
    T, N = 20, 100
    u0 = ad.array([1., 0.])
    t = linspace(0, T, N)
    u1 = integrate(lambda u : ad.hstack([u[1], -u[0]]), u0, t)
    u2 = integrate(lambda u : ad.hstack([u[1], -u[0]]), u0, [0, T])
    from pylab import *
    plot(t, ad.value(u1))
    plot([0, T], ad.value(u2), 'o')
