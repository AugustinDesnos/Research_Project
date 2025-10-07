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

import numpy as np
from unscented_transform import Gaussian, unscented_transform, plot_ellipse
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import *
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import ctypes
import multiprocessing as mp

lib = ctypes.CDLL("./integral.dll")
lib.integrand.argtypes = [
    ctypes.c_double,  # lat
    ctypes.c_double,  # lon
    ctypes.c_double,  # inc
    ctypes.c_int,     # N
    ctypes.c_double,  # mu_x
    ctypes.c_double,  # mu_y
    ctypes.c_double,  # sigma_x
    ctypes.c_double,  # sigma_y
    ctypes.c_double   # rho
]
lib.integrand.restype = ctypes.c_double

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
                                i, omega, raan, lv, PositionAngleType.TRUE,
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

    return pv, final_state, keplerian_orbit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def unwrap_angles(perigee):
    unwrapped = [perigee[0]]
    print(unwrapped[0])
    for i in range(1, len(perigee)):
        delta = perigee[i] - perigee[i-1]

        if delta < -np.pi:
            unwrapped.append(unwrapped[-1] + (perigee[i] + 2*np.pi - perigee[i-1]))
        elif delta > np.pi:
            unwrapped.append(unwrapped[-1] + (perigee[i] - 2*np.pi - perigee[i-1]))
        else:
            unwrapped.append(unwrapped[-1] + delta)
    print(unwrapped[0])
    return unwrapped

def plot_ellipse(mean: np.ndarray, cov: np.ndarray, ax, n_std=1, num_points=900, **kwargs):

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    theta = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.column_stack([
        ell_radius_x * np.cos(theta),
        ell_radius_y * np.sin(theta)
    ])

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    
    transformed_points = transf.transform(ellipse_points)

    ellipse_patch = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    ellipse_patch.set_transform(transf + ax.transData)
    ax.add_patch(ellipse_patch)

    return transformed_points

def inclination_model(alpha, i0, i1, i2):
    return i0 + i1 * np.cos(alpha) + i2 * np.cos(2 * alpha)

def raan_model(data, omega0, omega_dot, omega1, omega2):
    alpha, t = data
    return omega0 + omega_dot * t + omega1 * np.sin(alpha) + omega2 * np.sin(2 * alpha)

def raan_model_unwrapped(alpha, omega0, omega_alpha, omega1, omega2):
    return omega0 + omega_alpha*alpha + omega1 * np.sin(alpha) + omega2 * np.sin(2 * alpha)


def integrand_py(lon, lat, inc, mu_x, mu_y, sigma_x, sigma_y, rho):
    return lib.integrand(lat, lon, inc, 2, mu_x, mu_y, sigma_x, sigma_y, rho)

def integrate_cell(cell_lat, cell_lon, mean, cov):
    inc = np.radians(97.98)
    lat_min = np.radians(cell_lat)
    lat_max = np.radians(cell_lat + 1)
    lon_min = np.radians(cell_lon)
    lon_max = np.radians(cell_lon + 1)

    integral, _ = nquad(lambda lon, lat: integrand_py(lon, lat, inc, mean[0], mean[1], np.sqrt(cov[0,0]), np.sqrt(cov[1,1]),
                                                       cov[0,1]/(np.sqrt(cov[0,0])*np.sqrt(cov[1,1]))),
                 [[lon_min, lon_max], [lat_min, lat_max]])

    #print("Total Probability of the Integral =", integral)
    return integral


def total_casualty_risk(cells, mean, cov):
    total_risk = 0.0
    for lon_lat, pop_density in cells:
        #print(lon_lat)
        lon, lat = lon_lat
        if pop_density != 0 and lat<82.0:
            cell_integral = integrate_cell(lat, lon, mean, cov)
            #print("Population density =", pop_density)
            total_risk += pop_density * cell_integral
    return total_risk

if __name__ == "__main__":
    print("===================================================================================================================")
    print("Initializing the parameters for the start of the propagation")
    utc = TimeScalesFactory.getUTC()
    gmst = TimeScalesFactory.getGMST(IERSConventions.IERS_2010, False)

    epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
    initialDate = epochDate

    rp = 150000.0
    ra = 300000.0
    i = radians(98.2)
    omega = radians(0)
    raan = radians(0)
    lv = radians(0)
    satellite_mass = 90.0
    playerOne = [90, 1.1, 2.25]
    Airbus = [500, [2.78,7.41], 2.25]

    a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0    
    e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

    inertialFrame = FramesFactory.getEME2000()

    initialOrbit_0 = KeplerianOrbit(a, e, i, omega, raan, lv,
                                PositionAngleType.TRUE,
                                inertialFrame, epochDate, Constants.WGS84_EARTH_MU)

    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                            Constants.WGS84_EARTH_FLATTENING, 
                            ITRF)
    sun = CelestialBodyFactory.getSun()
    msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                                MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

    duration = 4*24*3600
    sigma_rp = 100.0
    sigma_ra = 1000.0
    sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
    sigma_k = 0.01
    mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
    covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
    initialgaussian = Gaussian(mean, covariance)


    sig_ae, gaussian_ae = sigma_points(mean, covariance)

    new_mean = np.zeros((4))
    new_mean[:2] = gaussian_ae.mean()
    new_mean[2] = 0.0
    new_mean[3] = 1
    new_covariance = np.zeros((4,4))
    new_covariance[:2, :2] = gaussian_ae.covariance()
    new_covariance[2, 2] = sigma_t**2
    new_covariance[3, 3] = sigma_k**2
    new_covariance[3,0] = 0
    new_covariance[0,3] = 0
    newgaussian = Gaussian(new_mean, new_covariance)
    sig_points = newgaussian.compute_sigma_points()

    pa = []
    inc = []
    LAN = []
    print("Initialization complete")
    print("===================================================================================================================")
    print("Start of propagation")

    for j in range(len(sig_points)):
        pvs, final_state, final_state_orbit = numerical_propagation(sig_points[j])
        if j == 1:
            og_final_state = final_state_orbit
        v = pvs.getVelocity()
        p = pvs.getPosition()
        t = initialDate.shiftedBy(float(final_state.getDate().durationFrom(initialDate)))
        RAAN = final_state_orbit.getRightAscensionOfAscendingNode()
        perig = final_state_orbit.getPerigeeArgument()
        incli = final_state_orbit.getI()
        trueano = final_state_orbit.getTrueAnomaly()
        pa.append(perig+trueano)
        gmstSeconds = final_state.getDate().getComponents(gmst).getTime().getSecondsInLocalDay()
        sid_time = (gmstSeconds / 240.0) % 360.0
        LAN.append(np.radians(np.degrees(RAAN) - sid_time))
        inc.append(np.degrees(incli))

    print("Propagation Complete")
    print("===================================================================================================================")

    LAN = np.array(LAN)
    pa = np.array(pa)
    inc = np.array(inc)

    sigma_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    sorted_indices = np.argsort(LAN)
    LAN_sorted = LAN[sorted_indices]
    pa_sorted = pa[sorted_indices]
    inc_sorted = inc[sorted_indices]
    reverse_lookup = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}
    sigma_sorted_indices = [reverse_lookup[i] for i in sigma_indices]
    unwrapped_pa = np.array(unwrap_angles(pa_sorted))

    final_sig_points = [(t, x) for t, x in zip(LAN_sorted[sigma_sorted_indices], unwrapped_pa[sigma_sorted_indices])]
    final_sig_points = np.array(final_sig_points)
    final_sig_points = final_sig_points[np.argsort(final_sig_points[:, 0])]

    final_mean = np.mean(final_sig_points, axis=0)
    final_covariance = np.cov(final_sig_points, rowvar=False)

    alpha_rad = unwrapped_pa
    indices = sigma_sorted_indices.copy()

    popt_i_unwrapped, _ = curve_fit(inclination_model, alpha_rad[sigma_sorted_indices], inc[:9])
    alpha_fit = np.linspace(min(alpha_rad), max(alpha_rad), 1000)
    i_fit = inclination_model(alpha_fit, *popt_i_unwrapped)

    ncols, nrows = 360, 145
    xllcorner, yllcorner = -180.0, -60.0
    cellsize = 1.0

    values = []
    with open("POP_100_2024_GPW_V4_UN_ext.asc", "r") as f:
        lines = f.readlines()[6:]
        for line in lines:
            values.extend(map(float, line.split()))

    values = np.array(values).reshape((nrows, ncols))

    longitudes = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
    latitudes = np.arange(yllcorner, yllcorner + nrows * cellsize, cellsize)
    latitudes = np.flip(latitudes)

    coords_with_values = [
        ([lon, lat], values[i, j])
        for i, lat in enumerate(latitudes)
        for j, lon in enumerate(longitudes)
        if values[i, j] > 0.0]

    inc = np.radians(97.98)
    rv = multivariate_normal(mean=final_mean, cov=final_covariance)

    mu_x = final_mean[0]
    mu_y = final_mean[1]
    sig_x = np.sqrt(final_covariance[0,0])
    sig_y = np.sqrt(final_covariance[1,1])
    num_procs = mp.cpu_count()  # use all available cores
    print("Starting integration")
    n_chunks = 12
    chunks = np.array_split(np.array(coords_with_values, dtype=object), n_chunks)
    args = [(chunk, final_mean, final_covariance) for chunk in chunks]
    with mp.Pool() as pool:
        results = pool.starmap(total_casualty_risk, args)
    print("Integration complete")
    total_result = sum(results)
    print("The probability of having a victim is", total_result/1e6)
