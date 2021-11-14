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

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission.load('mission_after_launch.pickle')
print('Look, I am still in space:', mission.rocket_launched)

'''Shortcut begin'''

code_stable_orbit = 95927
code_orientation = 43160
system = SolarSystem(seed)

shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit, code_orientation])

# Orientation software shortcut
pos, vel, angle = shortcut.get_orientation_data()
print("Position after launch:", pos)
print("Velocity after launch:", vel)

#Verifying orientation with shortcut data
mission.verify_manual_orientation(pos, vel, angle)

# Initialize interplanetary travel instance
travel = mission.begin_interplanetary_travel()

# Shortcut to make the landing sequence class start with a stable orbit
shortcut.place_spacecraft_in_stable_orbit(2, 500e3, 0, 6)

# Initializing landing sequence class instance
landing = mission.begin_landing_sequence()

# Calling landing sequece oreint function
t, pos, vel = landing.orient()

print("We are at:")
print("Time :", t)
print("pos :", pos)
print("vel :", vel)

'''Shortcut end'''
print('')
print('--------------------------')
print('')

'''
EGEN KODE
'''

landing.fall_until_time(600)
t, pos, vel = landing.orient()



print('')
print('FALLING')

print('Distance to the planet center:', np.linalg.norm(pos), 'm')

print('')
print('FALLING')

landing.fall_until_time(1200)
t, pos, vel = landing.orient()

print('')
print('FALLING')

print('Distance to the planet center:', np.linalg.norm(pos), 'm')

print('')

landing.look_in_direction_of_planet()
landing.start_video()
landing.fall(3600 * 2)
landing.finish_video()

t, pos, vel = landing.orient()

print('')
print('FALLING and recording video')

print('Distance to the planet center:', np.linalg.norm(pos), 'm')

'''
Challenge D
'''



def landing_site_coordinates(phi_coord, time_elapsed):
    '''Function to calculate new coordinates'''
    
    T = system.rotational_periods[6]
    omega = 2 * np.pi / T
    phi_new = phi_coord + omega * time_elapsed

    return phi_new
