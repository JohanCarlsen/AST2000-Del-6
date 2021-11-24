'''
Egen kode: Anton Brekke
'''

import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.shortcuts import SpaceMissionShortcuts
import scipy.constants as scs
from numba import njit

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission.load('mission_after_launch.pickle')
print('Look, I am still in space:', mission.rocket_launched)

"""
Data: [lambda, sigma, fluks]

sigma_noise_load = np.loadtxt('sigma_noise.txt')
spectrum_load = np.loadtxt('spectrum_seed97_600nm_3000nm.txt')[:, -1]

data = np.zeros((len(sigma_noise_load[:,0]), 3))
data[:,0] = sigma_noise_load[:,0]
data[:,1] = sigma_noise_load[:,1]
data[:,2] = spectrum_load[:]

np.save('data', data)
"""

lambda_data = np.load('lambda_data.npy')
sigma_noise = np.load('sigma_noise.npy')
F_data = np.load('F_data.npy')
print(f'smallest sigma noise:{np.min(sigma_noise)}')

c = const.c     # Lysfart SI
k = const.k_B       # Boltzmann
lambda0 = 820     # nm, laboratoriebølgelengde
m = 8*(scs.m_p + scs.m_n) + 2*scs.m_p
T = 300             # K, tempertur på gass
v_rel = 10000           # m/s, fart relativt til planet
print(v_rel / c)

d_lambda = v_rel / c * lambda0      # nm, Dopplerskift
print(d_lambda)
# print(d_lamda)
# print(d_lambda / 2)

# Regner ut standardavviket for profilen med bølgelengde lambda0
def sigma_calc(lambda0, m, T):
    return 2*lambda0 / c * np.sqrt(k*T / (4*m))

sigma = sigma_calc(lambda0, m, T)
print(sigma)

# Modell for fluksen, F(lambda)
@njit
def F(lambda_data, lambda0, sigma, Fmin):
    f = 1 + (Fmin - 1)*np.exp(-0.5*((lambda_data - lambda0) / sigma)**2)
    return f

# Indeks i intervall vi ønkser å se på
index_obs = np.logical_and(lambda_data >= lambda0 - d_lambda, lambda_data <= lambda0 + d_lambda)
plt.plot(lambda_data[index_obs], F_data[index_obs])
# plt.plot(lambda_data[index_obs], F(lambda_data[index_obs], lambda0, sigma, 0.7))
plt.xlabel(r'$\lambda$', weight='bold', fontsize=16)
plt.ylabel('F', weight='bold', fontsize=16)
plt.show()


n = 3
lambda_obs = 820.017        # Observert i datakurver
print(lambda_obs)
print(lambda_obs - n*sigma)
print(lambda_obs + n*sigma)
# Indeks rundt observert bølgelende vi er interessert i
index = np.logical_and(lambda_data <= lambda_obs + n*sigma, lambda_data >= lambda_obs - n*sigma)

# Implementerer X^2-metoden som skal minimere parametere
@njit
def X_squared(F_data, sigma_noise, lambda0, lambda_data, indexx):
    sigma_noisel = sigma_noise.copy()[indexx]
    F_datal = F_data.copy()[indexx]
    sigma_search = np.linspace(0, 3*sigma, len(F_datal))
    lambda_datal = lambda_data.copy()[indexx]
    lambda0l = np.linspace(lambda0 - n*sigma, lambda0 + n*sigma, len(F_datal))
    Fminl = np.linspace(0, 0.7, len(F_datal))
    X = np.zeros((len(F_datal), len(F_datal), len(F_datal)))
    print(X.shape)
    for i in range(len(F_datal)):
        for j in range(len(F_datal)):
            for k in range(len(F_datal)):
                # print(F(lambda_datal, lambda0l[i], sigma_noisel[j], Fminl[k]))
                X[i,j,k] = np.sum(((F_datal - F(lambda_datal, lambda0l[i], sigma_search[j], Fminl[k])) / sigma_noisel)**2)
    min = np.min(X)
    print(min)
    where = np.where(X==min)
    print(where)
    i, j, k = where
    return lambda0l[i], sigma_search[j], Fminl[k], min, i, j, k


lambda0, sigmaj, Fmin, min, i, j, k = X_squared(F_data, sigma_noise, lambda0, lambda_data, index)
print(lambda0, sigmaj, Fmin, min, (i,j,k))

# Prøver å plotte resultater
index = np.logical_and(lambda_data >= lambda0 - 100, lambda_data <= lambda0 + 100)
# plt.plot(lambda_data[index], F_data[index], color='tab:orange')
plt.plot(lambda_data[index], F_data[index], color='tab:blue')
plt.plot(lambda_data[index], F(lambda_data[index], lambda0, sigmaj, Fmin), linewidth=2, color='r')
plt.show()
