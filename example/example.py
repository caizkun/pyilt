# Term Project: Inverse Laplace Transform of Real-valued Relaxation Data
# @icaizk, Fall 2014

# Test examples
# call routine: ilt()

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.realpath('..'))

# import support routine
import ilt


# ------------------------------------------------------
# Example 1: Input two exponetial decays
# The analytical solution is two delta functions.
# ------------------------------------------------------

# define input
p = 0.5
z1 = 0.1
z2 = 1.
t_exp = np.logspace(-1, 2, 100)
F_exp = p*np.exp(-z1*t_exp) + (1. - p)*np.exp(-z2*t_exp)     # two exponentials

# define parameters
bound = np.array([0.01, 10])
Nz = 80
alpha = 1e-6

# compute ILT
z_exp, f_exp, res_lsq, res_reg = ilt.ilt(t_exp, F_exp, bound, Nz, alpha)

plt.figure()
plt.semilogx(t_exp, F_exp, 'bo-', label=r'input $F(t)$')
plt.semilogx(1./z_exp, z_exp*f_exp/5.0, 'ro-',
             label=r'output $zf(z) = \tau\rho(\tau)$')
plt.xlabel('$t$   or   $1/z = \\tau$', fontsize=18)
plt.legend(loc=0, frameon=False)
plt.savefig('example_2exp.pdf')


# ------------------------------------------------------
# Example 2: Input stretched exponential decay (KWW model)
# The analytical solution at beta = 0.5 is known.
# ------------------------------------------------------

# define input
z1 = 1.
beta = 0.5
t_kww = np.logspace(-2, 2, 100)
F_kww = np.exp(-(z1*t_kww)**beta)   # stretched exponential

# define parameters
bound = np.array([0.01, 1000])
Nz = 80
alpha = 1e-6

# compute ILT
z_kww, f_kww, res_lsq, res_reg = ilt.ilt(t_kww, F_kww, bound, Nz, alpha)

zf_kww_analy = 0.5*np.sqrt(z1/z_kww/np.pi)*np.exp(-0.25*z1/z_kww)

plt.figure()
plt.semilogx(t_kww, F_kww, 'bo-', label=r'input $F(t)$')
plt.semilogx(1./z_kww, z_kww*f_kww, 'ro-',
             label=r'output $zf(z) = \tau\rho(\tau)$')
plt.semilogx(1./z_kww, zf_kww_analy, 'g-',
             linewidth=2, label='analytial solution')
plt.xlim([1.e-2, 1.e2])
plt.xlabel('$t$   or   $1/z = \\tau$', fontsize=18)
plt.legend(loc=0, frameon=False)
plt.savefig('example_kww.pdf')


# ------------------------------------------------------
# Example 3: Input simulaition data at two temperatures
# Relaxation at high T is almost homogeneous, while at
# low T dynamic heterogeneity emerges.
# ------------------------------------------------------

# define function to read data files
def read_data(filename, start_line):
    """
    read data files
    """
    data = []
    with open(filename, 'r') as fid:
        count = 0
        for line in fid:
            if count < start_line:
                count += 1
                continue
            data.append(map(float, line.split('	')))
        # print data
    return np.array(data)

# load simulation data as input
file_nT = ['example_simu_highT.dat', 'example_simu_lowT.dat']
data_nT = []
data_nT.append(read_data(file_nT[0], 6))   # 1st column t, 2nd column F
data_nT.append(read_data(file_nT[1], 6))

# define parameters
bound = np.array([1.e-4, 1.e2])
Nz = 50
alpha = 4e-2

# compute ILT
z_highT, f_highT, res_lsq, res_reg = \
    ilt.ilt(data_nT[0][:, 0], data_nT[0][:, 1], bound, Nz, alpha)

z_lowT, f_lowT, res_lsq, res_reg = \
    ilt.ilt(data_nT[1][:, 0], data_nT[1][:, 1], bound, Nz, alpha)

plt.figure()
plt.semilogx(data_nT[0][:, 0], data_nT[0][:, 1], 'bo-', label=r'input $F(t)$')
plt.semilogx(1./z_highT, z_highT*f_highT, 'ro-',
             label=r'output $zf(z) = \tau\rho(\tau)$')
plt.xlabel('$t$ [ps]   or   $1/z = \\tau$ [ps]', fontsize=18)
plt.legend(loc=0, frameon=False)
plt.title('High Temperture Result')
plt.savefig('example_simuhighT.pdf')

plt.figure()
plt.semilogx(data_nT[1][:, 0], data_nT[1][:, 1], 'bo-', label=r'input $F(t)$')
plt.semilogx(1./z_lowT, z_lowT*f_lowT, 'ro-',
             label=r'output $zf(z) = \tau\rho(\tau)$')
plt.xlabel('$t$ [ps]   or   $1/z = \\tau$ [ps]', fontsize=18)
plt.legend(loc=0, frameon=False)
plt.title('Low Temperture Result')
plt.savefig('example_simulowT.pdf')
plt.show()
