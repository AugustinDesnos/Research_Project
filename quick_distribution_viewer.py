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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unscented_transform import Gaussian, unscented_transform

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

def numerical_propagation(x):
    initialOrbit = KeplerianOrbit(float(x[0]),
                                    float(x[1]),
                                float(x[6]), float(x[4]), raan, float(x[5]), PositionAngleType.TRUE,
                              inertialFrame, initialDate.shiftedBy(float(x[2])), Constants.WGS84_EARTH_MU)
    #set up the code for the numerical propagation ( use of integrator, tolerances)
    minStep = 0.0001
    maxstep = 1000.0
    initStep = 10.0

    positionTolerance = 0.01

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
    drag_model = IsotropicDrag(playerOne[1], float(x[3]*playerOne[2]))
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

    return final_state, keplerian_orbit

#set the initial date
utc = TimeScalesFactory.getUTC()
gmst = TimeScalesFactory.getGMST(IERSConventions.IERS_2010, False)

epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
initialDate = epochDate

#set our orbit/satellite parameters
rp = 150000.0
ra = 290000.0

i = radians(98.2)
omega = radians(0)
raan = radians(0)
lv = radians(0)
satellite_mass = 90.0
playerOne = [90, 1.1, 2.25]
Airbus = [500, [2.78,7.41], 2.25]

a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0    
e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

#set up the inertial frame where the satellite is defined
inertialFrame = FramesFactory.getEME2000()

#set up the celestial bodies for perturbations
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                         Constants.WGS84_EARTH_FLATTENING, 
                         ITRF)
sun = CelestialBodyFactory.getSun()
msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                              MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

duration = 5*24*3600
sigma_rp = 10.0
sigma_ra = 10.0
sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
sigma_k = 0.01
sigma_w = radians(0.05)
sigma_i = radians(0.01)
sigma_v = radians(0.05)
mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                  rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
initialgaussian = Gaussian(mean, covariance)

N = 100
rp_ra_samples = np.random.multivariate_normal(mean, covariance, size=N)

# Step 2: Transform each sample to (a, e)
a_samples = (rp_ra_samples[:, 0] + rp_ra_samples[:, 1]) / 2
e_samples = (rp_ra_samples[:, 0] - rp_ra_samples[:, 1]) / (rp_ra_samples[:, 1] + rp_ra_samples[:, 0])

ae_samples = np.vstack([a_samples, e_samples]).T

# Step 3: Estimate new mean and covariance
mu_ae = np.mean(ae_samples, axis=0)
covariance_ae = np.cov(ae_samples, rowvar=False)

num_samples = 3000

sig_rpra = initialgaussian.compute_sigma_points()
rpra_ellipse_samples = [initialgaussian.sample() for _ in range(num_samples)]
x_rpra = [item[0] for item in rpra_ellipse_samples]
y_rpra = [item[1] for item in rpra_ellipse_samples]


sig_ae, gaussian_ae = sigma_points(mean, covariance)

new_mean = np.zeros((7))
new_mean[:2] = gaussian_ae.mean()
new_mean[2] = 0.0
new_mean[3] = 1
new_mean[4] = omega
new_mean[5] = lv
new_mean[6] = i
new_covariance = np.zeros((7,7))
new_covariance[:2, :2] = gaussian_ae.covariance()
new_covariance[2, 2] = sigma_t**2
new_covariance[3, 3] = sigma_k**2
new_covariance[4, 4] = sigma_w**2
new_covariance[5, 5] = sigma_v**2
new_covariance[6, 6] = sigma_i**2
#print(new_covariance)
newgaussian = Gaussian(new_mean, new_covariance)
sig_points = newgaussian.compute_sigma_points()
print(sig_points[0])
weights = newgaussian.compute_weights()
samples = [newgaussian.sample() for _ in range(num_samples)]
#print(sig_points)
ae_ellipse_samples = [gaussian_ae.sample() for _ in range(num_samples)]
x_ae = [item[0] for item in ae_ellipse_samples]
y_ae = [item[1] for item in ae_ellipse_samples]

AOL = []
LAN = []

for j in range(1000):
    final_state, final_state_orbit = numerical_propagation(samples[j])
    RAAN = final_state_orbit.getRightAscensionOfAscendingNode()
    perig = final_state_orbit.getPerigeeArgument()
    trueano = final_state_orbit.getTrueAnomaly()
    AOL.append(perig+trueano)
    times = final_state.getDate().durationFrom(initialDate)/3600
    initial_sid_time = initialDate.getComponents(gmst).getTime().getSecondsInLocalDay()
    sid_time_diff = times*(360/24) + initial_sid_time/240
    LAN.append(RAAN - np.radians(sid_time_diff))

df = pd.DataFrame(LAN)
df.to_csv('LANdata.csv', index=False)
df2 = pd.DataFrame(AOL)
df2.to_csv('AOLdata.csv')
print(LAN)


plt.figure()
plt.scatter(LAN, AOL)
plt.grid()
plt.show()