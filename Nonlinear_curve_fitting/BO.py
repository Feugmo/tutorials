#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : Sep 23 6:23 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : BO.py
@Project : Nonlinear_curve_fitting
@Software: PyCharm
"""

import math
import torch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from torch.optim import SGD
from gpytorch.mlls import ExactMarginalLogLikelihood
# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
dtype = torch.float

vols = torch.tensor([13.71, 14.82, 16.0, 17.23, 18.52])

energies = torch.tensor([-56.29, -56.41, -56.46, -56.463, -56.41])


def Murnaghan(parameters, vol):
    """From Phys. Rev. B 28, 5480 (1983)"""
    E0, B0, BP, V0 = parameters

    E = E0 + B0 * vol / BP * (((V0 / vol) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1.0)

    return torch.tensor(E)




