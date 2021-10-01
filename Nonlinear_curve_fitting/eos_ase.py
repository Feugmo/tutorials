#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : Sep 23 3:11 p.m. 2021
@Author  : Conrard TETSASSI
@Email   : giresse.feugmo@gmail.com
@File    : eos_ase.py.py
@Project : Nonlinear_curve_fitting
@Software: PyCharm
"""

import numpy as np

from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT

from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState

a = 4.0  # approximate lattice constant
b = a / 2
ag = Atoms('Ag',
           cell=[(0, b, b), (b, 0, b), (b, b, 0)],
           pbc=1,
           calculator=EMT())  # use EMT potential
cell = ag.get_cell()
traj = Trajectory('Ag.traj', 'w')
for x in np.linspace(0.95, 1.05, 5):
    # for x in np.linspace(0.90, 0.95, 5):
    ag.set_cell(cell * x, scale_atoms=True)
    ag.get_potential_energy()
    traj.write(ag)

configs = read('Ag.traj@0:5')  # read 5 configurations
# Extract volumes and energies:
volumes = [ag.get_volume() for ag in configs]
energies = [ag.get_potential_energy() for ag in configs]

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print('Bulk modulus', B / kJ * 1.0e24, 'GPa')
print('lattice constant :',  v0**(1/3))
eos.plot('Ag_eos_1.png')
