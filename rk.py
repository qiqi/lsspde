# adarray.py defines adarray that emulates numpy.ndarray
# Copyright (C) 2014
# Qiqi Wang  qiqi.wang@gmail.com
# engineer-chaos.blogspot.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, print_function, absolute_import

import unittest
import numpy as np
import numpad as ad

def step(f, u0, dt):
    '''
    Diagonally Implicit Runge Kutta (DIRK) coefficients in Section 2.1 of
    Cash. IMA J Appl Math (1979) 24 (3): 293-301. doi: 10.1093/imamat/24.3.293
    '''
    class CashTable:
        a = np.polynomial.polynomial.polyroots([-1./6, 3./2, -3, 1])[1]
        t2 = (a**2 - 3./2*a + 1./3) / (a**2 - 2*a + 1./2)
        b1 = (t2/2 - 1./6) / ((t2 - a) * (1 - a))
        b2 = (a/2 - 1./6) / ((a - t2) * (1 - t2))
        c1 = (t2 - 1./2) / (t2 - a)
        c2 = (a - 1./2) / (a - t2)

    ff = ad.replace__globals__(f)

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


def pdeint(f, u0, t, relTol=1E-4, absTol=1E-6, obliviate=False, disp=0):
    '''
    To be used like ode23s, using numpad for Jacobian
    '''
    def _roundTo2ToK(n):
        log2n = np.log2(max(1, n))
        return 2**int(round(log2n))

    u = ad.array(u0).copy()
    uHistory = [u]
    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        iSubdiv, nSubdiv = 0, _roundTo2ToK((t[i+1] - t[i]) / dt)
        dt = (t[i+1] - t[i]) / nSubdiv
        while iSubdiv < nSubdiv:
            u3rd, u2nd = step(f, u, dt)
            uNorm = np.linalg.norm(ad.value(u))
            errNorm = np.linalg.norm(ad.value(u3rd) - ad.value(u2nd))
            if errNorm > max(absTol, relTol * uNorm):
                dt, iSubdiv, nSubdiv = 0.5 * dt, 2 * iSubdiv, 2 * nSubdiv
            else:
                iSubdiv += 1
                u = u3rd
                if obliviate:
                    u.obliviate()
                if disp:
                    print(t[i] + (t[i+1] - t[i]) * iSubdiv / nSubdiv)
                if errNorm < 0.25 * max(absTol, relTol * uNorm) and \
                        iSubdiv % 2 == 0 and nSubdiv > 1:
                    dt, iSubdiv, nSubdiv = 2 * dt, iSubdiv / 2, nSubdiv / 2
        assert iSubdiv == nSubdiv
        uHistory.append(u)
    return ad.array(uHistory)


class LSS(object):
    """
    Base class for both tangent and adjoint sensitivity analysis
    During __init__, a trajectory is computed,
    and the matrices used for both tangent and adjoint are built
    """
    def __init__(self, f, u0, s, T):
        self.f = f
        self.T = np.array(T, float).copy()
        self.s = np.array(s, float).copy()

        if self.s.ndim == 0:
            self.s = self.s[np.newaxis]

        u0 = np.array(u0, float)
        if u0.ndim == 1:
            # run up to T[0]
            f = lambda u : self.f(u, s)
            assert T[0] >= 0 and T.size > 1
            u0 = pdeint(f, u0, [0, T[0]], obliviate=True)[-1]
            # compute a trajectory
            self.u = pdeint(f, u0, T - T[0])
            np.save('.LSS.u.npy', ad.value(self.u))
        else:
            assert (u0.shape[0],) == T.shape
            self.u = u0.copy()

        self.dT = T[1:] - T[:-1]
        self.dudt = array([f(u) for u in self.u])


# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _RunThroughTest(unittest.TestCase):
    def testKuramotoSivashinsky(self):
        def kuramotoSivashinsky(u):
            dx = L / N
            uExt = np.hstack([0, u, 0])
            u2 = uExt**2
            u2x = (u2[2:] - u2[:-2]) / (4 * dx)
            uxx = (uExt[2:] + uExt[:-2] - 2 * uExt[1:-1]) / dx**2
            uxxExt = np.hstack([0, uxx, 0])
            uxxxx = (uxxExt[2:] + uxxExt[:-2] - 2 * uxxExt[1:-1]) / dx**2
            return -u2x - uxx - uxxxx
        L, N = 10., 16
        u0 = np.random.random(N-1)
        t = np.linspace(0, 10, 11)
        u = pdeint(kuramotoSivashinsky, u0, t, obliviate=True, disp=0)

class _GlobalErrorTest(unittest.TestCase):
    def testHarmonicOscillator(self):
        T, N = 10, 100
        u0 = ad.array([1., 0.])
        t = np.linspace(0, T, N)
        u1 = pdeint(lambda u : ad.hstack([u[1], -u[0]]), u0, t)
        u2 = pdeint(lambda u : ad.hstack([u[1], -u[0]]), u0, [0, T])
        accuracy = np.linalg.norm(ad.value(u1[-1] - u2[-1]))
        self.assertLess(accuracy, 5E-4)

if __name__ == '__main__':
    # unittest.main()
    def kuramotoSivashinsky(u):
        dx = L / N
        uExt = np.hstack([0, u, 0])
        u2 = uExt**2
        u2x = (u2[2:] - u2[:-2]) / (4 * dx)
        uxx = (uExt[2:] + uExt[:-2] - 2 * uExt[1:-1]) / dx**2
        uxxExt = np.hstack([0, uxx, 0])
        uxxxx = (uxxExt[2:] + uxxExt[:-2] - 2 * uxxExt[1:-1]) / dx**2
        return -u2x - uxx - uxxxx
    L, N = 100., 256
    u0 = np.random.random(N-1)
    LSS(kuramotoSivashinsky, u0, linspace(20, 20 + 256, 256 + 1))
