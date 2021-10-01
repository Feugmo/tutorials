#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : Sep 23 2:59 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : eos.py.py
@Project : Nonlinear_curve_fitting
@Software: PyCharm
"""
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import numpy as np

vols = np.array([13.71, 14.82, 16.0, 17.23, 18.52])

energies = np.array([-56.29, -56.41, -56.46, -56.463, -56.41])


def Murnaghan(parameters, vol):
    """From Phys. Rev. B 28, 5480 (1983)"""
    E0, B0, BP, V0 = parameters

    E = E0 + B0 * vol / BP * (((V0 / vol) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1.0)

    return E


def objective(pars, y, x):
    # we will minimize this function
    err = y - Murnaghan(pars, x)
    return err

# fit a parabola to the data and get inital guess for equilibirum volume
# and bulk modulus
a, b, c = np.polyfit(vols, energies, 2)
V0 = -b/(2*a)
E0 = a*V0**2 + b*V0 + c
B0 = 2*a*V0
Bp = 4.0

# initial guesses in the same order used in the Murnaghan function
x0 = [E0, B0, Bp, V0]

print('initial guesses [E0, B0, Bp, V0] = {}'.format(x0))

# x0 = [-56.0, 0.54, 2.0, 16.5]  # initial guess of parameters

plsq = leastsq(objective, x0, args=(energies, vols))

print('Fitted parameters = {0}'.format(plsq[0]))



plt.plot(vols, energies, 'ro')

# plot the fitted curve on top
x = np.linspace(min(vols), max(vols), 50)
y = Murnaghan(plsq[0], x)
plt.plot(x, y, 'k-')
plt.xlabel('Volume')
plt.ylabel('Energy')
plt.show()
plt.savefig('nonlinear-curve-fitting.png')

