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
import matplotlib.pyplot as plt

import numpy as np
from unscented_transform import Gaussian, unscented_transform
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import *
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import ctypes
import multiprocessing as mp
import time

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

#calculates the sigma points of the rp, ra Gaussian and converts them into semi-major axis (a) and eccentricity (e),
#then creates a new Gaussian.
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

#propagates the orbit from the initial orbit position. There is an altitude detector set up at 80km of altitude so that
#the propagator stops at that altitude, as this is where fragmentation of the satellite occurs.
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

#from the angles of the perigee and LAN that are periodic wrt the perigee (-2pi;2pi) and unvravels the perigee angle in
#in terms of the first two angles by assuming a straight line dependance between LAN and perigee
def unwrap_angles(perigee, lan):
    unwrapped = [perigee[0]]
    for i in range(1, len(perigee)):
        if i == 1:
            if perigee[i] > perigee[i-1]:
                unwrapped.append(unwrapped[-1] + (perigee[i] - 2*np.pi - perigee[i-1]))
            else:
                unwrapped.append(unwrapped[-1] + (perigee[i] - perigee[i-1]))
            alpha_grad = unwrapped[1] - unwrapped[0]
            lan_grad = lan[1] - lan[0]
            lan_period = lan_grad*2*np.pi/(-alpha_grad)
        else:
            if lan[i] - lan[i-1] < lan_period:
                if perigee[i] > perigee[i-1]:
                    unwrapped.append(unwrapped[-1] + (perigee[i] - 2*np.pi - perigee[i-1]))
                else:
                    unwrapped.append(unwrapped[-1] + (perigee[i] - perigee[i-1]))
            else:
                x = np.floor((lan[i]-lan[i-1])/lan_period)
                unwrapped.append(unwrapped[-1] + (perigee[i] - 2*x*np.pi - perigee[i-1]))
    return unwrapped

#the function guess for the inclination, to estimate the inclination of any point in terms of the alpha angle
def inclination_model(alpha, i0, i1, i2):
    return i0 + i1 * np.cos(alpha) + i2 * np.cos(2 * alpha)

#calls the integrand.dll file that was built from the C file integral.C
def integrand_py(lon, lat, inc, mu_x, mu_y, sigma_x, sigma_y, rho):
    return lib.integrand(lat, lon, inc, 5, mu_x, mu_y, sigma_x, sigma_y, rho)

#determines the longitude/latitude cell limits and calls the function integrand_py, then returns the result of that integration
def integrate_cell(cell_lat, cell_lon, mean, cov, inc):
    lat_min = np.radians(cell_lat)
    lat_max = np.radians(cell_lat + 1)
    lon_min = np.radians(cell_lon)
    lon_max = np.radians(cell_lon + 1)

    integral, _ = nquad(lambda lon, lat: integrand_py(lon, lat, inc, mean[0], mean[1], np.sqrt(cov[0,0]), np.sqrt(cov[1,1]),
                    cov[0,1]/(np.sqrt(cov[0,0])*np.sqrt(cov[1,1]))), [[lon_min, lon_max], [lat_min, lat_max]])

    #print("Total Probability of the Integral =", integral)
    return integral

#sums up each cell's integration times the population density at that cell, by calling the function integrate_cell
def total_casualty_risk(cells, mean, cov, inc):
    total_risk = 0.0
    for lon_lat, pop in cells:
        #print(lon_lat)
        lon, lat = lon_lat
        if pop != 0:
            cell_integral = integrate_cell(lat, lon, mean, cov, inc)
            #print("Population density =", pop_density)
            pop_density = pop/(12364*np.cos(radians(lat+0.5))*1e6)
            total_risk += pop_density * cell_integral
    return total_risk

#start of the main code
if __name__ == "__main__":
    #Initialize the parameters to enable the propagation of every sigma point
    start_time = time.time()
    print("===================================================================================================================")
    print("Initializing the parameters for the start of the propagation")
    utc = TimeScalesFactory.getUTC()
    gmst = TimeScalesFactory.getGMST(IERSConventions.IERS_2010, False)

    epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
    initialDate = epochDate
    gmstSeconds = initialDate.getComponents(gmst).getTime().getSecondsInLocalDay()
    t_sid = (gmstSeconds / 240.0) % 360.0

    rp = 150000.0
    ra = 300000.0
    i = radians(98.2)
    omega = radians(0)
    lv = radians(0)
    satellite_mass = 90.0
    playerOne = [90, 1.1, 2.25]
    Airbus = [500, [2.78,7.41], 2.25]

    a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0    
    e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

    inertialFrame = FramesFactory.getEME2000()
    
    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                            Constants.WGS84_EARTH_FLATTENING, 
                            ITRF)
    sun = CelestialBodyFactory.getSun()
    msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                                MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

    #creates the sigma points of the initial Gaussian by creating the initial covariance of the Gaussian
    duration = 4*24*3600
    sigma_rp = 100.0
    sigma_ra = 1000.0
    sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
    sigma_k = 0.01
    mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
    covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
    initialgaussian = Gaussian(mean, covariance)

    #transforms the rp, ra sigma points into a, e sigma points
    sig_ae, gaussian_ae = sigma_points(mean, covariance)

    #using the new sigma points, creates a new mean and new covariance that uses the other uncertainties in time (sigma_t) and in
    # the atmospheric density (sigma_k), to determine the sigma points that will be used in the propagation
    #Note this code determines sigma points twice: once to convert from rp, ra to a, e and another time to add the dimensions of time
    # and atmospheric density
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


    task_1_time = time.time()
    print("Initialization complete, time taken:", task_1_time - start_time, "seconds")
    lans = np.arange(0, 360, 10)
    lans = [360]
    casualty_risk = []
    for lan_value in lans:
        raan = radians(lan_value + t_sid)
        initialOrbit_0 = KeplerianOrbit(a, e, i, omega, raan, lv,
                                    PositionAngleType.TRUE,
                                    inertialFrame, epochDate, Constants.WGS84_EARTH_MU)
        #creates three new lists that will append their respective value for each propagated sigma point
        pa = []
        inc = []
        LAN = []
        print("===================================================================================================================")
        print("Start of propagation")

        #starts the propagation. Checks if the propagation takes too long, and uses the formula for LAN in terms of RAAN and sidereal
        #time to get the value of each sigma point's LAN after propagation.
        for j in range(len(sig_points)):
            pvs, final_state, final_state_orbit = numerical_propagation(sig_points[j])
            if j == 1:
                og_final_state = final_state_orbit
            v = pvs.getVelocity()
            p = pvs.getPosition()
            t = initialDate.shiftedBy(float(final_state.getDate().durationFrom(initialDate)))
            times = final_state.getDate().durationFrom(initialDate)/3600
            if times == duration/3600:
                raise SystemExit("Stopping: reentry takes longer than 4 days, impossible to estimate")
            RAAN = final_state_orbit.getRightAscensionOfAscendingNode()
            perig = final_state_orbit.getPerigeeArgument()
            incli = final_state_orbit.getI()
            trueano = final_state_orbit.getTrueAnomaly()
            pa.append(perig+trueano)
            gmstSeconds = final_state.getDate().getComponents(gmst).getTime().getSecondsInLocalDay()
            sid_time = (gmstSeconds / 240.0) % 360.0
            LAN.append(np.radians(np.degrees(RAAN) - sid_time))
            inc.append(np.degrees(incli))

        task_2_time = time.time()
        print("Propagation Complete, time taken:", task_2_time - task_1_time, "seconds")
        print("===================================================================================================================")

        LAN = np.array(LAN)
        pa = np.array(pa)
        inc = np.array(inc)

        #sorts all the lists in terms of the LAN sorting (similar to time sorting), to allow the unwrap_angles function to work. 
        # Note the sigma_indices list is put here in case samples are propagated as well as the sigma points.
        sigma_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        sorted_indices = np.argsort(LAN)
        LAN_sorted = LAN[sorted_indices]
        pa_sorted = pa[sorted_indices]
        inc_sorted = inc[sorted_indices]
        reverse_lookup = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}
        sigma_sorted_indices = [reverse_lookup[i] for i in sigma_indices]
        unwrapped_pa = np.array(unwrap_angles(pa_sorted, LAN_sorted))

        #inputs the resulting sigma points in 2D only: LAN and unwrapped alpha angles (alpha = perigee here)
        #Note in the case where the initial Gaussian is in 4D and we wish to only view the transformed Gaussian in 2D, we determine
        #the resulting covariance by taking all the points into account for the covariance.
        final_sig_points = [(t, x) for t, x in zip(LAN_sorted[sigma_sorted_indices], unwrapped_pa[sigma_sorted_indices])]
        final_sig_points = np.array(final_sig_points)
        final_sig_points = final_sig_points[np.argsort(final_sig_points[:, 0])]

        #Determines the resulting 2D Gaussian, defined by the mean and the covariance
        final_mean = np.mean(final_sig_points, axis=0)
        final_covariance = np.cov(final_sig_points, rowvar=False)

        alpha_rad = unwrapped_pa
        indices = sigma_sorted_indices.copy()

        #uses the guess function for the inlination to obtain a function of the inclination in terms of the alpha angle
        popt_i_unwrapped, _ = curve_fit(inclination_model, alpha_rad[sigma_sorted_indices], inc[:9])
        alpha_fit = np.linspace(min(alpha_rad), max(alpha_rad), 1000)
        i_fit = inclination_model(alpha_fit, *popt_i_unwrapped)

        #creates all the cells for the POP_100_2024_etc... file that is a population density file.
        ncols, nrows = 360, 145
        xllcorner, yllcorner = -180.0, -60.0
        cellsize = 1.0

        #transforms the file so that it appends only the longitudes, latitudes and population density in the file.
        # the list should look like this: values = [[(-180.0, 85.0), 0.0],[(-179.0, 85.0), 0.0], etc...] 
        values = []
        with open("POP_100_2024_GPW_V4_UN_ext.asc", "r") as f:
            lines = f.readlines()[6:]
            for line in lines:
                values.extend(map(float, line.split()))

        values = np.array(values).reshape((nrows, ncols))

        #creates a list of longitudes and latitudes and flips the latitudes file as the start of latitudes is at 85 degrees, but the
        #np.arange function doesn't like using yllcorner - nrows * cellsize if yllcorner is at 85 degrees.
        longitudes = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
        latitudes = np.arange(yllcorner, yllcorner + nrows * cellsize, cellsize)
        latitudes = np.flip(latitudes)

        #removes all the values of the list that have a population density of 0, and takes the mean value of the inclination
        #obtained from the inclination_fit function
        #Note the inclination is needed to convert the latitudes and longitudes into (alpha, LAN) in order to determine the probability
        #of a point having these coordinates in the Gaussian space of (alpha, LAN).
        coords_with_values = [
            ([lon, lat], values[i, j])
            for i, lat in enumerate(latitudes)
            for j, lon in enumerate(longitudes)
            if values[i, j] > 0.0]
        inc = popt_i_unwrapped[0]

        num_procs = mp.cpu_count()
        start_int_time = time.time()
        
        print("Starting integration")
        #separates the coords_with_values list into 12 as it is the number of processors this laptop has, and starts the integration
        #by calling the total_casualty_risk function through pool.starmap, and adding as arguments, the coords_with_values, the final_mean
        #and covariance of the transformed Gaussian, and the mean inclination
        total_result = total_casualty_risk(coords_with_values, final_mean, final_covariance,inc)
        #n_chunks = 12
        #chunks = np.array_split(np.array(coords_with_values, dtype=object), n_chunks)
        #args = [(chunk, final_mean, final_covariance, inc) for chunk in chunks]
        #with mp.Pool() as pool:
        #    results = pool.starmap(total_casualty_risk, args)
        end_int_time = time.time()
        print("Integration complete, time taken:", end_int_time - start_int_time, "seconds")
        #casualty_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(playerOne[1]))**2
        #total_result = sum(results)
        casualty_risk.append(total_result)
        print("The probability of having a victim is", total_result)
    plt.figure()
    plt.plot(lans, casualty_risk, label = 'ra = 300km, rp = 150km')
    plt.title('Casualty risk for set apogee and perigee in terms of the LAN angle')
    plt.xlabel('LAN angle (in degrees)')
    plt.ylabel('Casualty risk')
    plt.grid(True)
    plt.show()