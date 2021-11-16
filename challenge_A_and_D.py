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

print('')
print('FALLING')

landing.fall_until_time(600)
t, pos, vel = landing.orient()

print('')
print('Distance to the planet center:', np.linalg.norm(pos), 'm')
print('Speed:', np.linalg.norm(vel), 'm/s')

print('')
print('FALLING')

landing.fall_until_time(1200)
t, pos, vel = landing.orient()

print('')
print('Distance to the planet center:', np.linalg.norm(pos), 'm')
print('Speed:', np.linalg.norm(vel), 'm/s')

print('')
print('FALLING and recording video')

landing.look_in_direction_of_planet()
# landing.start_video()
landing.fall(3600 * 2)
# landing.finish_video()

t, pos, vel = landing.orient()

print('')
print('Distance to the planet center:', np.linalg.norm(pos), 'm')
print('Speed:', np.linalg.norm(vel), 'm/s')


'''
Challenge D
'''

radius = system.radii[6] * 1000     # planet radius in meters


def landing_site_coordinates(coords, time_elapsed):
    '''
    Function to calculate new coordinates.
    To get from cartesian to spherical coords:

    x = rho sin(theta) cos(phi)
    y = rho sin(theta) sin(phi)
    z = rho cos(theta)

    where

        0 <   rho     <= infinity
        0 <=  theta   <= pi
        0 <=  phi     <= 2pi
    '''

    x, y, z = coords

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan(y / x)
    theta = np.arccos(z / rho)

    T = system.rotational_periods[6]
    omega = 2 * np.pi / T

    phi_new = phi + omega * time_elapsed
    rho_new = radius

    x_new = rho_new * np.sin(theta) * np.cos(phi)
    y_new = rho_new * np.sin(theta) * np.sin(phi)
    z_new = rho_new * np.cos(theta)

    new_coords = np.array([x_new, y_new, z_new])

    return new_coords

print('')

T_one_orbit = np.ceil(2 * np.pi * np.linalg.norm(pos) / np.linalg.norm(vel))    # approximate orbit time

print(f'Time it takes for spacecraft to complete one orbit: {T_one_orbit} s')
'''
Time it takes for spacecraft to complete one orbit: 6875.0 s
'''

print('')
print('---------------------------------------------')
print('')
print('SEARCH FOR LANDING SPOT')
print('')

for dt in range(0, int(T_one_orbit), 860):
    landing.fall_until_time(t + dt)
    t_new, pos_new, vel_new = landing.orient()
    # landing.take_picture(f'search_for_landing_spot_at_time_{int(t_new)}.xml')
    new_coords = landing_site_coordinates(pos_new, T_one_orbit)
    print('')
    print(f'Picture taken at time {t_new} s has new coordinates:\n{new_coords} m\nafter {T_one_orbit} s')
    print('')

'''
SEARCH FOR LANDING SPOT

Spacecraft fell until time 8400 s.
Performed automatic orientation:
Time: 8400 s
Position: (415822, 2.32494e+06, 0) m
Velocity: (-2124.82, 380.129, 0) m/s
XML file search_for_landing_spot_at_time_8400.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_8400.xml.

Picture taken at time 8400.0 s has new coordinates:
[3.27776756e+05 1.83266244e+06 1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 9260 s.
Performed automatic orientation:
Time: 9260 s
Position: (-1.35107e+06, 1.93733e+06, 0) m
Velocity: (-1770.51, -1234.64, 0) m/s
XML file search_for_landing_spot_at_time_9260.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_9260.xml.

Picture taken at time 9260.0 s has new coordinates:
[ 1.06496445e+06 -1.52706907e+06  1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 10120 s.
Performed automatic orientation:
Time: 10120 s
Position: (-2.32551e+06, 413360, 0) m
Velocity: (-377.76, -2125.13, 0) m/s
XML file search_for_landing_spot_at_time_10120.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_10120.xml.

Picture taken at time 10120.0 s has new coordinates:
[ 1.83301157e+06 -3.25818658e+05  1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 10980 s.
Performed automatic orientation:
Time: 10980 s
Position: (-1.93598e+06, -1.35305e+06, 0) m
Velocity: (1236.53, -1769.16, 0) m/s
XML file search_for_landing_spot_at_time_10980.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_10980.xml.

Picture taken at time 10980.0 s has new coordinates:
[1.52598755e+06 1.06651357e+06 1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 11840 s.
Performed automatic orientation:
Time: 11840 s
Position: (-410935, -2.32585e+06, 0) m
Velocity: (2125.62, -375.458, 0) m/s
XML file search_for_landing_spot_at_time_11840.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_11840.xml.

Picture taken at time 11840.0 s has new coordinates:
[3.23919039e+05 1.83334821e+06 1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 12700 s.
Performed automatic orientation:
Time: 12700 s
Position: (1.35514e+06, -1.93434e+06, 0) m
Velocity: (1767.87, 1238.61, 0) m/s
XML file search_for_landing_spot_at_time_12700.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_12700.xml.

Picture taken at time 12700.0 s has new coordinates:
[ 1.06822218e+06 -1.52479198e+06  1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 13560 s.
Performed automatic orientation:
Time: 13560 s
Position: (2.32622e+06, -408081, 0) m
Velocity: (372.97, 2126.17, 0) m/s
XML file search_for_landing_spot_at_time_13560.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_13560.xml.

Picture taken at time 13560.0 s has new coordinates:
[ 1.83374114e+06 -3.21687199e+05  1.13998917e-10] m
after 6875.0 s

Spacecraft fell until time 14420 s.
Performed automatic orientation:
Time: 14420 s
Position: (1.93259e+06, 1.35758e+06, 0) m
Velocity: (-1240.76, 1766.39, 0) m/s
XML file search_for_landing_spot_at_time_14420.xml was saved in XMLs/.
It can be viewed in MCAst.
Picture saved to search_for_landing_spot_at_time_14420.xml.

Picture taken at time 14420.0 s has new coordinates:
[1.52343344e+06 1.07015876e+06 1.13998917e-10] m
after 6875.0 s
'''
