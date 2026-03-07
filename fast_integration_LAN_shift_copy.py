# The Gaussian class and unscented_transform used in this code are from:
# https://github.com/hugohadfield/unscented_transform

#setup all the imports needed for the code
import orekit
vm = orekit.initVM()
from math import radians, degrees
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
from scipy.interpolate import RectBivariateSpline


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

from juliacall import Main as jl
from juliacall import Pkg as jlPkg
import pandas as pd


#sets up Julia and the package CasualtyRisk, in order to compute the casualty risk of a fallout area.
jlPkg.activate("CasualtyRisk") # Replace with relative path to the CasualtyRisk folder
jlPkg.instantiate()
jl.seval("using CasualtyRisk")

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

    return pv, final_state, keplerian_orbit


#Uses the list of points with coordinates (LAN, alpha) to unwrap the points so that instead of looking periodic, they
# follow a straight line. It uses a best-fit gradient that was determined using a Monte-Carlo propagation. 
def unwrap_angles(perigee, lan):
    perigee = np.array(perigee)
    lan = np.array(lan)
    
    # Save the sort mask so we know how the original points moved
    sort = np.argsort(lan)
    lan = lan[sort]
    perigee = perigee[sort]

    mean_index = int(np.floor(len(perigee)/2))
    factor = (np.abs(lan[mean_index])//(2*np.pi))
    for j in range(len(lan)):
        lan[j] = lan[j] + factor*2*np.pi
        
    best_grad = -16.834325684236013
    
    line_perig = (lan*best_grad - lan[mean_index]*best_grad + perigee[mean_index] + np.pi)%(2*np.pi) - np.pi
    shifts = perigee - line_perig
    
    for j in range(len(shifts)):
        if np.abs(shifts[j]) > np.pi:
            if perigee[j] < perigee[j-1]:
                shifts[j] = -(np.abs(shifts[j]) - 2*np.pi)
            else:
                shifts[j] = np.abs(shifts[j]) - 2*np.pi
                
    new_perigee = perigee[mean_index] + (lan*best_grad - lan[mean_index]*best_grad + perigee[mean_index]) + shifts - perigee[mean_index]
    
    # Return the sort mask!
    return new_perigee, lan, sort

#the function guess for the inclination, to estimate the inclination of any point in terms of the alpha angle
def inclination_model(alpha, i0, i1, i2):
    return i0 + i1 * np.cos(alpha) + i2 * np.cos(2 * alpha)

#code to determine the size of the casualty area required for a random reentry to have a casualty risk of
#1e-4. The data set used was taken from CNES: guide_des_bonnes_pratiques_los_satellites_-_draft.pdf
def random_risk(year, inclination, spacecraft):
    data = {
    2021: [9.06, 9.7, 6.85, 6.4, 6.64, 4.95, 4.74, 4.36, 4.69, 5.3, 5.79, 6.16, 6.84, 7.52, 8.01, 8.37, 8.62, 8.77, 8.82, 8.81, 8.79, 8.75, 8.69, 8.62, 8.54, 8.43, 8.31, 8.17, 8.01],
    2022: [8.83, 9.52, 6.73, 6.29, 6.56, 4.89, 4.69, 4.32, 4.65, 5.25, 5.75, 6.12, 6.78, 7.45, 7.94, 8.3, 8.55, 8.69, 8.74, 8.73, 8.71, 8.67, 8.62, 8.55, 8.46, 8.36, 8.24, 8.1, 7.94],
    2023: [8.61, 9.34, 6.61, 6.18, 6.47, 4.83, 4.64, 4.27, 4.6, 5.2, 5.7, 6.07, 6.73, 7.39, 7.87, 8.23, 8.47, 8.62, 8.66, 8.66, 8.63, 8.6, 8.54, 8.47, 8.39, 8.28, 8.17, 8.03, 7.87],
    2024: [8.4, 9.16, 6.49, 6.07, 6.38, 4.77, 4.59, 4.23, 4.56, 5.16, 5.65, 6.02, 6.67, 7.32, 7.8, 8.15, 8.4, 8.54, 8.59, 8.58, 8.56, 8.52, 8.46, 8.4, 8.31, 8.21, 8.09, 7.96, 7.8],
    2025: [8.18, 8.99, 6.37, 5.97, 6.29, 4.71, 4.54, 4.19, 4.51, 5.11, 5.6, 5.97, 6.61, 7.26, 7.73, 8.08, 8.32, 8.46, 8.51, 8.5, 8.48, 8.44, 8.39, 8.32, 8.23, 8.13, 8.02, 7.88, 7.73],
    2030: [7.26, 8.19, 5.84, 5.49, 5.9, 4.46, 4.32, 4.01, 4.32, 4.9, 5.4, 5.77, 6.38, 6.99, 7.44, 7.77, 8, 8.13, 8.18, 8.17, 8.15, 8.11, 8.06, 8, 7.92, 7.82, 7.71, 7.58, 7.44],
    2035: [6.47, 7.49, 5.35, 5.07, 5.56, 4.24, 4.13, 3.85, 4.15, 4.73, 5.24, 5.6, 6.18, 6.76, 7.19, 7.5, 7.72, 7.85, 7.89, 7.89, 7.87, 7.83, 7.78, 7.72, 7.65, 7.55, 7.45, 7.33, 7.19],
    2040: [5.79, 6.85, 4.91, 4.69, 5.24, 4.06, 3.97, 3.72, 4.01, 4.57, 5.09, 5.45, 6.01, 6.56, 6.97, 7.27, 7.48, 7.61, 7.65, 7.64, 7.62, 7.59, 7.54, 7.48, 7.41, 7.32, 7.22, 7.11, 6.97],
    2045: [5.21, 6.28, 4.5, 4.36, 4.96, 3.9, 3.83, 3.61, 3.89, 4.43, 4.97, 5.33, 5.86, 6.4, 6.79, 7.08, 7.28, 7.4, 7.44, 7.43, 7.41, 7.38, 7.34, 7.28, 7.21, 7.13, 7.03, 6.92, 6.79],
    2050: [4.72, 5.76, 4.13, 4.05, 4.71, 3.76, 3.7, 3.51, 3.8, 4.31, 4.87, 5.23, 5.74, 6.26, 6.63, 6.91, 7.11, 7.22, 7.26, 7.25, 7.24, 7.2, 7.16, 7.11, 7.04, 6.96, 6.86, 6.76, 6.63],
    2055: [4.24, 5.28, 3.79, 3.76, 4.45, 3.61, 3.57, 3.41, 3.68, 4.18, 4.75, 5.11, 5.6, 6.09, 6.45, 6.72, 6.91, 7.02, 7.05, 7.05, 7.03, 7, 6.96, 6.91, 6.84, 6.76, 6.67, 6.57, 6.45],
    2060: [3.82, 4.83, 3.48, 3.49, 4.21, 3.46, 3.44, 3.3, 3.57, 4.05, 4.64, 5, 5.46, 5.93, 6.28, 6.54, 6.72, 6.82, 6.86, 6.85, 6.84, 6.81, 6.77, 6.72, 6.65, 6.58, 6.49, 6.39, 6.28],
    2065: [3.44, 4.43, 3.19, 3.24, 3.99, 3.33, 3.32, 3.2, 3.47, 3.93, 4.53, 4.89, 5.33, 5.78, 6.11, 6.36, 6.53, 6.64, 6.67, 6.67, 6.65, 6.62, 6.58, 6.53, 6.47, 6.4, 6.32, 6.22, 6.11],
    2070: [3.09, 4.06, 2.93, 3.01, 3.77, 3.19, 3.2, 3.11, 3.37, 3.81, 4.42, 4.78, 5.2, 5.64, 5.95, 6.19, 6.36, 6.45, 6.49, 6.48, 6.47, 6.44, 6.4, 6.36, 6.3, 6.23, 6.15, 6.06, 5.95],
    2075: [2.79, 3.72, 2.69, 2.79, 3.57, 3.07, 3.08, 3.01, 3.27, 3.7, 4.31, 4.67, 5.08, 5.49, 5.8, 6.02, 6.18, 6.28, 6.31, 6.3, 6.29, 6.26, 6.23, 6.18, 6.13, 6.06, 5.98, 5.9, 5.8],
    2080: [2.51, 3.41, 2.47, 2.59, 3.38, 2.95, 2.97, 2.92, 3.17, 3.58, 4.21, 4.57, 4.96, 5.35, 5.64, 5.86, 6.01, 6.11, 6.14, 6.13, 6.12, 6.09, 6.06, 6.01, 5.96, 5.9, 5.82, 5.74, 5.64],
    2085: [2.26, 3.12, 2.26, 2.41, 3.2, 2.83, 2.87, 2.83, 3.08, 3.48, 4.11, 4.47, 4.84, 5.21, 5.49, 5.7, 5.85, 5.94, 5.97, 5.96, 5.95, 5.93, 5.89, 5.85, 5.8, 5.74, 5.67, 5.59, 5.49],
    2090: [2.03, 2.86, 2.07, 2.23, 3.03, 2.72, 2.76, 2.75, 2.99, 3.37, 4.01, 4.37, 4.72, 5.08, 5.35, 5.55, 5.69, 5.78, 5.8, 5.8, 5.79, 5.76, 5.73, 5.69, 5.64, 5.58, 5.51, 5.44, 5.35],
    2095: [1.83, 2.62, 1.9, 2.07, 2.86, 2.61, 2.66, 2.67, 2.9, 3.27, 3.92, 4.27, 4.61, 4.95, 5.21, 5.4, 5.54, 5.62, 5.64, 5.64, 5.63, 5.61, 5.57, 5.54, 5.49, 5.43, 5.37, 5.29, 5.21],
    2100: [1.65, 2.4, 1.75, 1.92, 2.71, 2.51, 2.57, 2.59, 2.81, 3.17, 3.82, 4.18, 4.5, 4.83, 5.07, 5.25, 5.38, 5.46, 5.49, 5.48, 5.47, 5.45, 5.42, 5.38, 5.34, 5.28, 5.22, 5.15, 5.07]
    }
    inclinations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110]
    years = sorted(data.keys())
    values = np.array([data[y] for y in years])
    interp_func = RectBivariateSpline(years, inclinations, values, s=0)
    cas_area = float(interp_func(year, inclination)[0][0])
    sat_cas_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(spacecraft[1]))**2

    return (1e-4)*sat_cas_area/cas_area

def final_mean_covariance(perigee, lan, sort_mask, n):
    mean_perig = 0
    mean_lan = 0
    covariance = np.zeros((2,2))
    
    # Create weights in the ORIGINAL Unscented Transform order
    # (Assuming the UT library generates the central mean point at index 0)
    original_weights = np.array([1/(2*(n*2 - 3)) for _ in range(len(perigee))])
    original_weights[0] = (n - 3)/(n*2 - 3)
    
    # Sort the weights to perfectly match the scrambled perigee and lan arrays
    sorted_weights = original_weights[sort_mask]
    
    # Compute the mean:
    for i in range(len(perigee)):
        mean_perig += sorted_weights[i] * perigee[i]
        mean_lan += sorted_weights[i] * lan[i]
        
    # Compute the covariance matrix:
    for i in range(len(perigee)):
        covariance[0, 0] += sorted_weights[i] * (lan[i] - mean_lan)**2
        covariance[0, 1] += sorted_weights[i] * (perigee[i] - mean_perig) * (lan[i] - mean_lan)
        covariance[1, 1] += sorted_weights[i] * (perigee[i] - mean_perig)**2
        
    covariance[1, 0] = covariance[0, 1]
    
    return np.array([mean_lan, mean_perig]), covariance

def total_risk(risk_dict, pop_dict):
    """
    Calculates the total expected casualty risk by matching 
    coordinates between the risk map and the population map.
    """
    total = 0.0
    
    # We only iterate over populated cells to save time
    for coords, pop in pop_dict.items():
        # If that exact (lat, lon) exists in our Julia results, multiply them
        if coords in risk_dict:
            total += pop * risk_dict[coords]
            
    return total

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
    t_sid = (gmstSeconds / 240.0)

    rp_values = range(150000, 170000, 100000)
    ra_values = range(300000, 360000, 100000)

    initial_inc = radians(98.2)
    omega = radians(0)
    lv = radians(0)
    satellite_mass = 90.0
    playerOne = [90, 1.1, 2.25]
    Airbus = [500, [2.78,7.41], 2.25]

    parameter_sets = [{'rp': x, 'ra': y} for x in rp_values for y in ra_values]

    # This list will hold all of our data before we save it
    all_results = []

    for idx, params in enumerate(parameter_sets):
    # Create a dictionary to store the row data for THIS specific run
        row_data = {'run_id': idx}
        row_data.update(params) # Adds param_A and param_B to the row
        a = (params['rp'] + params['ra'] + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0    
        e = 1.0 - (params['rp'] + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

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
        sigma_rp = 500.0
        sigma_ra = 1000.0
        sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
        sigma_k = 0.1
        sigma_w = radians(0.5)
        sigma_i = radians(0.01)
        sigma_v = radians(0.5)
        mean = np.array([params['ra']+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                        params['rp']+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
        covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
        initialgaussian = Gaussian(mean, covariance)

        #transforms the rp, ra sigma points into a, e sigma points
        sig_ae, gaussian_ae = sigma_points(mean, covariance)

        #using the new sigma points, creates a new mean and new covariance that uses the other uncertainties in time (sigma_t) and in
        # the atmospheric density (sigma_k), to determine the sigma points that will be used in the propagation
        #Note this code determines sigma points twice: once to convert from rp, ra to a, e and another time to add the dimensions of time
        # and atmospheric density
        new_mean = np.zeros((7))
        new_mean[:2] = gaussian_ae.mean()
        new_mean[2] = 0.0
        new_mean[3] = 1
        new_mean[4] = omega
        new_mean[5] = lv
        new_mean[6] = initial_inc
        new_covariance = np.zeros((7,7))
        new_covariance[:2, :2] = gaussian_ae.covariance()
        new_covariance[2, 2] = sigma_t**2
        new_covariance[3, 3] = sigma_k**2
        new_covariance[4, 4] = sigma_w**2
        new_covariance[5, 5] = sigma_v**2
        new_covariance[6, 6] = sigma_i**2
        newgaussian = Gaussian(new_mean, new_covariance)
        sig_points = newgaussian.compute_sigma_points()


        
        #print("Initialization complete, time taken:", task_1_time - start_time, "seconds")
        initial_lan_value = 0
        J_list = []
        risks = []
        #n_chunks = 12
        #chunks = np.array_split(np.array(coords_with_values, dtype=object), n_chunks)
        #args = [(chunk, final_mean, final_covariance, inc) for chunk in chunks]
        #with mp.Pool() as pool:
        rp = params['rp']
        ra = params['ra']
        print(f'Starting at altitudes {rp/1000}km and {ra/1000}km')
        omega_values = np.arange(0, 360, 1000)
        for omega in omega_values:
            new_mean[4] = omega
            print(f'Perigee Argument: {omega}')
            raan = radians(initial_lan_value + t_sid)
            #creates three new lists that will append their respective value for each propagated sigma point
            pa = []
            inc = []
            LAN = []
            #print("===================================================================================================================")
            #print("Start of propagation")

            #starts the propagation. Checks if the propagation takes too long, and uses the formula for LAN in terms of RAAN and sidereal
            #time to get the value of each sigma point's LAN after propagation.
            task_1_time = time.time()
            for j in range(len(sig_points)):
                pvs, final_state, final_state_orbit = numerical_propagation(sig_points[j])
                if j == 1:
                    og_final_state = final_state_orbit
                v = pvs.getVelocity()
                p = pvs.getPosition()
                t = initialDate.shiftedBy(float(final_state.getDate().durationFrom(initialDate)))
                times = final_state.getDate().durationFrom(initialDate)/3600
                if times == duration/3600:
                    raise SystemExit("Stopping: reentry takes longer than 4 days")
                RAAN = final_state_orbit.getRightAscensionOfAscendingNode()
                perig = final_state_orbit.getPerigeeArgument()
                incli = final_state_orbit.getI()
                trueano = final_state_orbit.getTrueAnomaly()
                pa.append(perig+trueano)
                initial_sid_time = initialDate.getComponents(gmst).getTime().getSecondsInLocalDay()
                sid_time_diff = times*(360/24) + initial_sid_time/240
                LAN.append(np.radians(np.degrees(RAAN) - sid_time_diff))
                inc.append(np.degrees(incli))
            task_2_time = time.time()
            print("Propagation Complete, time taken:", task_2_time - task_1_time, "seconds")
            #print("===================================================================================================================")

            LAN = np.array(LAN)
            pa = np.array(pa)
            inc = np.array(inc)
            plt.figure()
            plt.scatter(LAN, pa)
            plt.grid()
            plt.show()
            #sorts all the lists in terms of the LAN sorting (similar to time sorting), to allow the unwrap_angles function to work. 
            # Note the sigma_indices list is put here in case samples are propagated as well as the sigma points.
            unwrapped_pa, lan_unwrapped, sort_idx = unwrap_angles(pa, LAN)
            unwrapped_pa = np.array(unwrapped_pa)
            lan_unwrapped = np.array(lan_unwrapped)
            plt.figure()
            plt.scatter(lan_unwrapped, unwrapped_pa)
            plt.grid()
            plt.show()


            #Determines the resulting 2D Gaussian, defined by the mean and the covariance
            final_mean, final_covariance = final_mean_covariance(unwrapped_pa, lan_unwrapped, sort_idx, 7)
            print(f'Final mean = {final_mean}')
            print(f'Final Covariance = {final_covariance}')
            alpha_rad = unwrapped_pa

            row_data['final_mean'] = final_mean
            # Flatten the covariance matrix into separate columns
            row_data['final_cov_00'] = final_covariance[0, 0]
            row_data['final_cov_01'] = final_covariance[0, 1]
            row_data['final_cov_10'] = final_covariance[1, 0]
            row_data['final_cov_11'] = final_covariance[1, 1]


            #uses the guess function for the inlination to obtain a function of the inclination in terms of the alpha angle
            popt_i_unwrapped, _ = curve_fit(inclination_model, alpha_rad, inc)
            print(f'The inclination fit found that the mean inclination is {popt_i_unwrapped[0]}')

            #creates all the cells for the POP_100_2024_GPW_V4_UN_ext.asc file that is a population density file.
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

            longitudes = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
            latitudes = np.arange(yllcorner, yllcorner + nrows * cellsize, cellsize)
            latitudes = np.flip(latitudes)
            GRID_SIZE = 1

            pop_dict = {}
            for i, lat in enumerate(latitudes):
                for j, lon in enumerate(longitudes):
                    pop = values[i, j]     
                    if pop > 0.0:
                        coord_key = (np.round(lat, 1), np.round(lon, 1))
                        pop_dict[coord_key] = pop
            inc = radians(popt_i_unwrapped[0])

            cellsize = 1.0
            GRID_SIZE = 1.0
            cell_longitudes = np.arange(-180.0, 180.0, cellsize)
            cell_latitudes = np.arange(-90.0, 90.0, cellsize)

            cells = [(radians(lat), radians(lat + GRID_SIZE), radians(lon), radians(lon + GRID_SIZE))
                    for lat in cell_latitudes
                    for lon in cell_longitudes]
            casualty_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(playerOne[1]))**2
            int_start = time.time()
            integration_results = jl.compute_casualty_risk(cells, final_mean, final_covariance, inc, casualty_area, 1e-3)

            risk_matrix = np.array(integration_results).reshape((180, 360))

            pop_targets = []
            for (lat, lon), pop in pop_dict.items():
                row = int(round(lat + 90.0)) 
                col = int(round(lon + 180.0))
                if 0 <= row < 180 and 0 <= col < 360:
                    pop_targets.append((row, col, pop))

            random_reentry = random_risk(2020, degrees(inc), playerOne)
            print(f'The risk for a random reentry at the same inclination is: {random_reentry}')

            risk_list = []

            for shift in range(360):
                shifted_matrix = np.roll(risk_matrix, shift, axis=1)
                total = 0.0
                for r, c, pop in pop_targets:
                    total += pop * shifted_matrix[r, c]
                total_risk_value = total * casualty_area
                
                risk_list.append(total_risk_value / random_reentry)

            for degree, val in enumerate(risk_list):
                row_data[f'risk_{degree}_degrees'] = val
            final_time = time.time() - int_start
            print('Integration time taken:', final_time, 'seconds')

            min_value = np.min(risk_list)
            min_index = np.argmin(risk_list)
            row_data['min_value_and_index'] = min_value
            row_data['min_degree'] = min_index
            all_results.append(row_data)

            lans = np.arange(0, 360, 1)
            plt.figure()
            plt.plot(lans, risk_list, label = 'ra = 300km, rp = 150km')
            plt.title('J(rp, ra) in terms of the LAN angle')
            plt.xlabel('LAN angle (in degrees)')
            plt.ylabel('J(rp, ra)')
            plt.grid(True)
            plt.show()
            print(f'The minimum is found at LAN = {np.argmin(np.array(risk_list))}, with a value of {risk_list[np.argmin(np.array(risk_list))]}')
            J_list.append((risk_list[np.argmin(np.array(risk_list))], omega, np.argmin(np.array(risk_list))))
            risks.append(risk_list[np.argmin(np.array(risk_list))])
            print(f'The mean of all the risks is: {np.mean(risk_list)}')
    df = pd.DataFrame(all_results)
    df.to_csv('my_experiment_results.csv', index=False)

    print("Data successfully saved! Shape of data:", df.shape)
    print(f'The list of all the minimum values in terms of the perigee angle is: {J_list}')
    print(f'Total time taken: {time.time() - start_time} seconds')
    #plt.figure()
    #plt.plot(omega_values, risks, label = 'ra = 300km, rp = 150km')
    #plt.xlabel('Perigee Argument (in degrees)')
    #plt.ylabel('J(rp, ra)')
    #plt.grid(True)

    #plt.show()
