#%%
# The Gaussian class and unscented_transform used in this code are from:
# https://github.com/hugohadfield/unscented_transform

#setup all the imports needed for the code
import orekit
vm = orekit.initVM()
from math import radians
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()
from org.orekit.orbits import OrbitType, KeplerianOrbit, PositionAngleType
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.frames import FramesFactory
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.events import AltitudeDetector
from org.orekit.propagation.events.handlers import StopOnEvent
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from orekit import JArray_double
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from unscented_transform import Gaussian, unscented_transform, plot_ellipse
from scipy.optimize import curve_fit
from scipy import interpolate


#set the initial date
utc = TimeScalesFactory.getUTC() 
epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
initialDate = epochDate
print(Constants.WGS84_EARTH_ANGULAR_VELOCITY)
#set our orbit/satellite parameters
rp = 150000.0
ra = 300000.0
i = radians(98.2)
omega = radians(0)
raan = radians(0.0)
lv = radians(0)
satellite_mass = 90.0
playerOne = [90, 1.1, 2.25]
Airbus = [500, [2.78,7.41], 2.25]

a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0    
e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

#set up the inertial frame where the satellite is defined
inertialFrame = FramesFactory.getEME2000()

#set up the orbit construction as Keplerian
initialOrbit_0 = KeplerianOrbit(a, e, i, omega, raan, lv,
                              PositionAngleType.TRUE,
                              inertialFrame, epochDate, Constants.WGS84_EARTH_MU)

#set up the Earth for the plotting of the orbit later on
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                         Constants.WGS84_EARTH_FLATTENING, 
                         ITRF)
sun = CelestialBodyFactory.getSun()
msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                              MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

#the propagation of the initial sigma points into the semi-major axis and the eccentricity
def sigma_points(mean, covariance):
    def non_linear_function(x):
        a = (x[0] + x[1]) / 2.0
        e = 1.0 - (x[1]) / a
        return np.array([a, e])
    
    gaussian = Gaussian(mean, covariance)
    
    transformed_gaussian = unscented_transform(gaussian, non_linear_function)
    
    sigma_points = gaussian.compute_sigma_points()
    transformed_sigma_points = [non_linear_function(sp) for sp in sigma_points]
    return transformed_sigma_points, transformed_gaussian

#the propagation function
def numerical_propagation(x):
    initialOrbit = KeplerianOrbit(float(x[0]),
                                    float(x[1]),
                                i, omega, raan, lv, PositionAngleType.TRUE,
                              inertialFrame, initialDate.shiftedBy(float(x[2])), Constants.WGS84_EARTH_MU)
    #set up the code for the numerical propagation ( use of integrator, tolerances)
    minStep = 0.0001
    maxstep = 1000.0
    initStep = 10.0

    positionTolerance = 1.0

    tolerances = NumericalPropagator.tolerances(positionTolerance, 
                                                initialOrbit, 
                                                initialOrbit.getType())

    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tolerances[0]),
        JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(initStep)

    #set up the propagator
    initialState = SpacecraftState(initialOrbit, satellite_mass) 
    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(initialState)
    altitude_detector = AltitudeDetector(80000.0, earth).withHandler(StopOnEvent())
    propagator_num.addEventDetector(altitude_detector)
    atmosphere = NRLMSISE00(msafe, sun, earth)
    drag_model = IsotropicDrag(playerOne[1], playerOne[2])
    drag_force = DragForce(atmosphere, drag_model)
    propagator_num.addForceModel(drag_force)

    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))
    finalDate = initialDate.shiftedBy(float(duration))
    final_state = propagator_num.propagate(initialDate, finalDate)
    pv = final_state.getPVCoordinates()
    OrbitType.CARTESIAN.convertType(propagator_num.getInitialState().getOrbit())
    orbit = final_state.getOrbit()
    keplerian_orbit = KeplerianOrbit(orbit)

    return pv, final_state, keplerian_orbit

duration = 3*24*3600
sigma_rp = 100.0
sigma_ra = 100.0
sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                  rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
sig_ae, gaussian_ae = sigma_points(mean, covariance)

new_mean = np.zeros((3))
new_mean[:2] = gaussian_ae.mean()
new_mean[2] = 0.0
new_covariance = np.zeros((3,3))
new_covariance[:2, :2] = gaussian_ae.covariance()
new_covariance[2, 2] = sigma_t**2

newgaussian = Gaussian(new_mean, new_covariance)
sig_points = newgaussian.compute_sigma_points()
weights = newgaussian.compute_weights()
num_samples = 1000
samples = [newgaussian.sample() for _ in range(num_samples)]
samples[:0] = sig_points

#plot the initial semi major axis and eccentricity Gaussian covariance ellipse 
#(haven't managed to add in the random samples aswell but will change that soon)
plt.figure()
ax = plt.gca()
plot_ellipse(gaussian_ae.mean(), gaussian_ae.covariance(), ax, n_std=1, facecolor='none', edgecolor='green')  
plt.scatter([x[0] for x in sig_ae], [x[1] for x in sig_ae], color='green')
plt.tight_layout()
plt.show()

lat = []
lon = []
pa = []
time = []
positions = []

#propagate each sample and get the parameters needed
for j in range(len(samples)):
    pvs, final_state, final_state_orbit = numerical_propagation(samples[j])
    p = pvs.getPosition()
    x = p.getX()
    y = p.getY()
    z = p.getZ()
    position = [x, y, z]
    positions.append(position)
    t = initialDate.shiftedBy(float(final_state.getDate().durationFrom(initialDate)))
    perig = final_state_orbit.getPerigeeArgument()
    trueano = final_state_orbit.getTrueAnomaly()
    pa.append(np.degrees(perig+trueano))
    time.append(final_state.getDate().durationFrom(initialDate)/3600)
    print(final_state_orbit)
    subpoint = earth.transform(p, inertialFrame, t)
    lat.append(np.degrees(subpoint.getLatitude()))
    lon.append(np.degrees(subpoint.getLongitude()))

#gaussian function (was used previously to show gaussian dependance of the perigee angles
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

#perigee angle unwrapping function
def unwrap_angles(perigee):
    unwrapped = [perigee[0]]

    for i in range(1, len(perigee)):
        delta = perigee[i] - perigee[i-1]

        if delta < -180:
            unwrapped.append(unwrapped[-1] + (perigee[i] + 360 - perigee[i-1]))
        elif delta > 180:
            unwrapped.append(unwrapped[-1] + (perigee[i] - 360 - perigee[i-1]))
        else:
            unwrapped.append(unwrapped[-1] + delta)

    return unwrapped

time = np.array(time)
pa = np.array(pa)
positions = np.array(positions)

#unwrap the perigee angles to avoid periodic relation with time
sigma_indices = [0, 1, 2, 3, 4, 5, 6]
sorted_indices = np.argsort(time)
time_sorted = time[sorted_indices]
pa_sorted = pa[sorted_indices]
reverse_lookup = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}
sigma_sorted_indices = [reverse_lookup[i] for i in sigma_indices]
print(sigma_sorted_indices)
unwrapped_pa = np.array(unwrap_angles(pa_sorted))

#plot the perigee angles in terms of the time it took the satellite to reach 80km of altitude
plt.figure()
plt.scatter(time[7:], pa[7:], color = "orange", alpha = 0.3, marker = "+")
plt.scatter(time[:7], pa[:7], color = "red")
plt.title("Perigee in terms of time of propagation")
plt.grid(True)
plt.show()

#plot the unwrapped angles to show the gaussian resemblance
plt.figure()
plt.scatter(time_sorted, unwrapped_pa, color = "orange", alpha = 0.3, marker = "+")
plt.scatter(time_sorted[sigma_sorted_indices], unwrapped_pa[sigma_sorted_indices], color = "red")
plt.title("Scattered unwrapped perigees in terms of time")
plt.grid(True)
plt.show()

#calulate the new mean and variance of the sigma points
final_sig_points = [(t, x) for t, x in zip(time_sorted[sigma_sorted_indices], unwrapped_pa[sigma_sorted_indices])]
final_sig_points = np.array(final_sig_points)
final_sig_points = final_sig_points[np.argsort(final_sig_points[:, 0])]
final_sig_points = np.delete(final_sig_points, [1, 5], axis=0)

final_mean = np.mean(final_sig_points, axis=0)
final_covariance = np.cov(final_sig_points, rowvar=False)

#plot the fitted Gaussian ellipse
plt.figure()
ax = plt.gca()
plot_ellipse(final_mean, final_covariance, ax, n_std=1, facecolor='none', edgecolor='green')  
plt.scatter(time_sorted, unwrapped_pa, marker = "+", color = "orange", alpha = 0.3)
plt.title("Gaussian ellipse fitted to it")
plt.tight_layout()
plt.show()

#create 3D plot to visualize where in its orbit does the satellite arrive at 80km of altitude
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2], label = "Orbit Trajectory")
u, v = np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100)
U, V = np.meshgrid(u, v)
x_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS*np.cos(U)*np.sin(V)
y_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS*np.sin(U)*np.sin(V)
z_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS*np.cos(V)
ax.plot_surface(x_earth, y_earth, z_earth, color = "blue", alpha = 0.3)
for lati in np.linspace(-80, 80, 9):
    lat_rad = np.radians(lati)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.cos(theta) * np.cos(lat_rad)
    y = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.sin(theta) * np.cos(lat_rad)
    z = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.sin(lat_rad)
    ax.plot(x, y, z, color='k', linewidth=0.3)

for long in np.linspace(0, 360, 12):
    lon_rad = np.radians(long)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 100)
    x = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.cos(phi) * np.cos(lon_rad)
    y = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.cos(phi) * np.sin(lon_rad)
    z = Constants.WGS84_EARTH_EQUATORIAL_RADIUS * np.sin(phi)
    ax.plot(x, y, z, color='k', linewidth=0.3)
ax.set_title("Satellite Points of Entry")
ax.set_box_aspect([1,1,1])
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
ax.set_xlim(np.min(positions[:,:3]),np.max(positions[:,:3]))
ax.set_ylim(np.min(positions[:,:3]),np.max(positions[:,:3]))
ax.set_zlim(np.min(positions[:,:3]),np.max(positions[:,:3]))
plt.show()

#create the map plot to visualize landing
m = Basemap(projection='cyl',
            resolution='l',
            area_thresh=None)

m.drawmapboundary()
m.fillcontinents(color='#dadada', lake_color='white')
m.drawmeridians(np.arange(-180, 180, 30), color='gray')
m.drawparallels(np.arange(-90, 90, 30), color='gray')
m.scatter(lon, lat, s=10, alpha=1, color='red', zorder=3, marker='.')

    # %%
