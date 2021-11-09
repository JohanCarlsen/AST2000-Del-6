'''
SNARVEI OG EGEN KODE
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

'''Shortcut begin'''

# code_stable_orbit = 95927
# code_orientation = 43160
# system = SolarSystem(seed)
#
# shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit, code_orientation])
#
# # Orientation software shortcut
# pos, vel, angle = shortcut.get_orientation_data()
# print("Position after launch:", pos)
# print("Velocity after launch:", vel)
#
# #Verifying orientation with shortcut data
# mission.verify_manual_orientation(pos, vel, angle)
#
# # Initialize interplanetary travel instance
# travel = mission.begin_interplanetary_travel()
#
# # Shortcut to make the landing sequence class start with a stable orbit
# shortcut.place_spacecraft_in_stable_orbit(0, 1000e3, 0, 1)
#
# # Initializing landing sequence class instance
# landing = mission.begin_landing_sequence()
#
# # Calling landing sequece oreint function
# t, pos, vel = landing.orient()
#
# print("We are at:")
# print("Time :", t)
# print("pos :", pos)
# print("vel :", vel)

'''Shortcut end'''

'''
EGEN KODE
'''
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


"""
Kan slice hvert hundrede, data[]
"""

lambda_data = np.load('lambda_data.npy')
sigma_noise = np.load('sigma_noise.npy')
F_data = np.load('F_data.npy')


c = const.c
k = const.k_B
lambda0 = 632      # nm
m = 8*(scs.m_p + scs.m_n)
T = 350             # K
v_rel = 10000           # m/s

d_lambda = v_rel / c * lambda0      # nm
# print(d_lamda)
# print(d_lambda / 2)

def sigma_calc(lambda0, m, T):
    return 2*lambda0 / c * np.sqrt(k*T / (4*m))

sigma = sigma_calc(lambda0, m, T)
print(sigma)

@njit
def F(lambda_data, lambda0, sigma, Fmin):
    f = 1 + (Fmin - 1)*np.exp(-0.5*((lambda_data - lambda0) / sigma)**2)
    return f

index_obs = np.logical_and(lambda_data >= lambda0 - d_lambda, lambda_data <= lambda0 + d_lambda)
plt.plot(lambda_data[index_obs], F_data[index_obs])
plt.show()


n = 10
lambda_obs = 362.019
print(lambda_obs - n*sigma)
print(lambda_obs + n*sigma)
index = np.logical_and(lambda_data <= lambda_obs + n*sigma, lambda_data >= lambda_obs - n*sigma)
print(index[index])
# plt.plot(lambda_data[index], F_data[index], color='tab:orange')
# plt.plot(lambda_data[index], F(lambda_data[index], lambda0, sigma, 0.7), color='tab:blue')
# plt.plot(lambda_data[index], (F_data[index] - F(lambda_data[index], lambda0, sigma, 0.7))**2)

sigma_noise = sigma_noise.copy()[index]
F_data = F_data.copy()[index]
lambda_data = lambda_data.copy()[index]

# @njit
def X_squared(F_data, sigma_noise, lambda0, lambda_data):
    sigma_noisel = sigma_noise.copy()
    F_datal = F_data.copy()
    lambda_datal = lambda_data.copy()
    lambda0l = np.linspace(lambda0 - d_lambda - n*sigma, lambda0 + d_lambda + n*sigma, len(F_datal))
    Fminl = np.linspace(0, 0.7, len(F_datal))
    X = np.zeros((len(F_datal), len(F_datal), len(F_datal)))
    print(X.shape)
    for i in range(len(F_datal)):
        for j in range(len(F_datal)):
            for k in range(len(F_datal)):
                X[i,j,k] = np.sum(((F_datal - F(lambda_datal, lambda0l[i], sigma_noisel[j], Fminl[k])) / sigma_noisel)**2, axis=0)
                # print(X[i,j,k])
    min = np.nanmin(X)
    where = np.where(X==min)
    print(where)
    i, j, k = where
    # print(where[0][50:100])
    # print(min, where, X[where])
    return lambda0_list, sigma_list, Fmin_list, min_list, index_list


lambda0, sigmaj, Fmin, min, ijk = X_squared(F_data, sigma_noise, lambda0, lambda_data)
print(lambda0, sigmaj, Fmin, min, ijk)
# fig = plt.figure()
# ax = fig.add_subplot(1,2,1, projection='3d')
# ax2 = fig.add_subplot(1,2,2)
# I = np.linspace(0, len(Fmin)-1, len(Fmin))
# rx, ry = np.meshgrid(I, I)
# for m in range(len(Fmin)):
#     ax.set_zlim(np.max(X[:,:,:]))
#     ax.plot_surface(rx, ry, X[:,:,m].T, cmap='jet')
#     ax2.contourf(rx, ry, X[:,:,m].T, levels=25, cmap='jet')
#     plt.pause(0.1)
#     ax.cla(); ax2.cla()

index = np.logical_and(lambda_data >= lambda0[0] - d_lambda, lambda_data <= lambda0[0] + d_lambda)
# plt.plot(lambda_data[index], F_data[index], color='tab:orange')
plt.plot(lambda_data, F(lambda_data, lambda0[0], sigmaj[0], Fmin[0]), linewidth=2, color='royalblue')
plt.show()
