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
    ad.replace_func_globals(f)
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
    ad.restore_func_globals(f)
    return u3, u_second_order

def roundTo2ToK(n):
    log2n = log2(max(1, n))
    return 2**int(round(log2n))

def integrate(f, u0, t, relTol=1E-4, absTol=1E-6, args=(), argv={}):
    u = ad.array(u0).copy()
    uHistory = [u]
    dt = t[1] - t[0]
    for i in range(len(t) - 1):
        nSubdiv = roundTo2ToK((t[i+1] - t[i]) / dt)
        dt = (t[i+1] - t[i]) / nSubdiv
        j = 0
        while j < nSubdiv:
            u3rd, u2nd = step(f, u, dt, args, argv)
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

# =========================================================== #
#                                                             #
#                         unittests                           #
#                                                             #
# =========================================================== #

class _RunThroughTest(unittest.TestCase):
    def testKuramotoSivashinsky(self):
        pass

class _GlobalErrorTest(unittest.TestCase):
    def testHarmonicOscillator(self):
        T, N = 10, 100
        u0 = ad.array([1., 0.])
        t = linspace(0, T, N)
        u1 = integrate(lambda u : ad.hstack([u[1], -u[0]]), u0, t)
        u2 = integrate(lambda u : ad.hstack([u[1], -u[0]]), u0, [0, T])
        accuracy = np.linalg.norm(ad.value(u1[-1] - u2[-1]))
        self.assertLess(accuracy, 5E-4)

if __name__ == '__main__':
    # unittest.main()
    def kuramotoSivashinsky(u, dx):
        uExt = hstack([0, u, 0])
        u2 = uExt**2
        u2x = (u2[2:] - u2[:-2]) / (4 * dx)
        uxx = (uExt[2:] + uExt[:-2] - 2 * uExt[1:-1]) / dx**2
        uxxExt = hstack([0, uxx, 0])
        uxxxx = (uxxExt[2:] + uxxExt[:-2] - 2 * uxxExt[1:-1]) / dx**2
        return -u2x - uxx - uxxxx

    L, N = 100., 100
    u0 = np.random.random(N-1)
    t = linspace(0, 100, 101)
    u = integrate(kuramotoSivashinsky, u0, t, args=(L / N,))
