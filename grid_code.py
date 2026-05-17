import sys
import time
import numpy as np
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# The Gaussian class and unscented_transform used in this code are from:
# https://github.com/hugohadfield/unscented_transform
from unscented_transform import Gaussian, unscented_transform

# --- GLOBAL OREKIT INITIALIZATION ---
import orekit
vm = orekit.initVM() 
from math import radians, degrees
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.orbits import Orbit, OrbitType, KeplerianOrbit, PositionAngleType
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

# -------------------------------------------------------------------------
# Standalone Numerical Propagation
# -------------------------------------------------------------------------

def numerical_propagation(sp, raan, i, omega, lv, initial_date, duration, mass, drag_area, drag_coeff):
    """
    Executes the orbital propagation. 
    Returns ONLY primitive Python floats to prevent pickling errors across processes.
    """
    inertialFrame = FramesFactory.getEME2000()
    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, ITRF)
    
    current_date = initial_date.shiftedBy(float(sp[2]))
    
    initialOrbit = KeplerianOrbit(float(sp[0]), float(sp[1]), float(sp[6]), float(sp[4]), float(raan), float(sp[5]),
                                  PositionAngleType.TRUE, inertialFrame, current_date, Constants.WGS84_EARTH_MU)
    
    # Explicit cast to bypass the spawned JCC ambiguity
    orbit_cast = Orbit.cast_(initialOrbit)
    tolerances = NumericalPropagator.tolerances(0.01, orbit_cast, orbit_cast.getType())
    
    integrator = DormandPrince853Integrator(0.0001, 1000.0, JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(10.0)

    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(SpacecraftState(initialOrbit, float(mass)))
    propagator_num.addEventDetector(AltitudeDetector(80000.0, earth).withHandler(StopOnEvent()))
    
    sun = CelestialBodyFactory.getSun()
    msafe = MarshallSolarActivityFutureEstimation(MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
                                                  MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
    drag_model = IsotropicDrag(float(drag_area), float(sp[3] * drag_coeff))
    propagator_num.addForceModel(DragForce(NRLMSISE00(msafe, sun, earth), drag_model))
    
    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(ITRF, gravityProvider))
    
    final_state = propagator_num.propagate(initial_date, initial_date.shiftedBy(float(duration)))
    
    # Cast Cartesian state back to Keplerian to extract angles safely
    keplerian_orbit = KeplerianOrbit(final_state.getOrbit())
    
    RAAN_val = keplerian_orbit.getRightAscensionOfAscendingNode()
    perig_val = keplerian_orbit.getPerigeeArgument()
    incli_val = keplerian_orbit.getI()
    trueano_val = keplerian_orbit.getTrueAnomaly()
    elapsed_time = final_state.getDate().durationFrom(initial_date)
    
    # Return strictly pure Python numbers
    return float(perig_val + trueano_val), float(RAAN_val), float(incli_val), float(elapsed_time)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def load_population_targets(filename):
    print(f"Loading population matrix from {filename}...")
    try:
        ncols, nrows = 360, 145
        xllcorner, yllcorner = -180.0, -60.0
        values = []
        with open(filename, "r") as f:
            for line in f.readlines()[6:]:
                values.extend(map(float, line.split()))
        values = np.array(values).reshape((nrows, ncols))
        longitudes = np.arange(xllcorner, xllcorner + ncols * 1.0, 1.0)
        latitudes = np.flip(np.arange(yllcorner, yllcorner + nrows * 1.0, 1.0))

        pop_targets = []
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                if values[i, j] > 0.0:
                    row = int(round(lat + 90.0))  
                    col = int(round(lon + 180.0)) 
                    if 0 <= row < 180 and 0 <= col < 360:
                        pop_targets.append((row, col, values[i, j]))
        return pop_targets
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

# Computes the sigma points of the (a, e) probability distribution using the uncertainty in rp and ra.
def sigma_points(mean, covariance):
    def non_linear_function(x):
        a = (x[0] + x[1]) / 2.0
        e = 1.0 - (x[1]) / a
        return np.array([a, e])
    gaussian = Gaussian(mean, covariance)
    transformed_gaussian = unscented_transform(gaussian, non_linear_function)
    return [non_linear_function(sp) for sp in gaussian.compute_sigma_points()], transformed_gaussian

# Unwrapping function: from the sigma points, all wrapped between -2pi and 2pi, it is necessary to unwrap them so that they go from -inf to inf.
# The samples all followed a straight line function, with gradient = -16.308802503037768. By taking this gradient, it is then possible
# to map the sigma points onto the straight line fit, taking into account the slight error from the straight line.
# IF the uncertainties are much more precise, it is possible to create a much simpler code.
def unwrap_angles(perigee, lan):
    perigee, lan = np.array(perigee), np.array(lan)
    sort = np.argsort(lan)
    lan, perigee = lan[sort], perigee[sort]

    mean_index = int(np.floor(len(perigee)/2))
    factor = (np.abs(lan[mean_index])//(2*np.pi))
    for j in range(len(lan)): lan[j] += factor*2*np.pi
        
    best_grad = -16.308802503037768
    line_perig = (lan*best_grad - lan[mean_index]*best_grad + perigee[mean_index] + np.pi)%(2*np.pi) - np.pi
    shifts = perigee - line_perig
    
    for j in range(len(shifts)):
        if np.abs(shifts[j]) > np.pi:
            shifts[j] = -(np.abs(shifts[j]) - 2*np.pi) if perigee[j] < perigee[j-1] else np.abs(shifts[j]) - 2*np.pi
                
    new_perigee = perigee[mean_index] + (lan*best_grad - lan[mean_index]*best_grad + perigee[mean_index]) + shifts - perigee[mean_index]
    return new_perigee, lan, sort

# Not a necessary function as we simply take the average inclination to compute the casualty risk.
def inclination_model(alpha, i0, i1, i2):
    return i0 + i1 * np.cos(alpha) + i2 * np.cos(2 * alpha)

# Uses a table obtained from the CNES to determine the value of the random risk associated with a satellite falling into the atmosphere.
# Inputs: date, inclination of the orbit, and spacecraft area (we consider here that the spacecraft does not fragment)
def random_risk(year, inclination, spacecraft_area):
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
    return (1e-4) * ((np.sqrt((0.677/2)**2*np.pi) + np.sqrt(spacecraft_area))**2) / cas_area

# Uses the definition of the unscented transform to determine the mean and covariance of the propagated distribution using the sigma points
def final_mean_covariance(perigee, lan, sort_mask, n):
    mean_perig, mean_lan, covariance = 0, 0, np.zeros((2,2))
    original_weights = np.array([1/(2*(n*2 - 3)) for _ in range(len(perigee))])
    original_weights[0] = (n - 3)/(n*2 - 3)
    sorted_weights = original_weights[sort_mask]
    
    for i in range(len(perigee)):
        mean_perig += sorted_weights[i] * perigee[i]
        mean_lan += sorted_weights[i] * lan[i]
        
    for i in range(len(perigee)):
        covariance[0, 0] += sorted_weights[i] * (lan[i] - mean_lan)**2
        covariance[0, 1] += sorted_weights[i] * (perigee[i] - mean_perig) * (lan[i] - mean_lan)
        covariance[1, 1] += sorted_weights[i] * (perigee[i] - mean_perig)**2
        
    covariance[1, 0] = covariance[0, 1]
    return np.array([mean_lan, mean_perig]), covariance

# -------------------------------------------------------------------------
# Worker Logic
# -------------------------------------------------------------------------

def process_omega_chunk(omega_chunk, rp, ra, pop_targets, static_cells, duration, playerOne):
    import traceback
    from juliacall import Main as jl
    from juliacall import Pkg as jlPkg
    
    jlPkg.activate("CasualtyRisk") 
    jlPkg.instantiate()
    jl.seval("using CasualtyRisk")

    if not orekit.getVMEnv():
        vm.attachCurrentThread()

    utc = TimeScalesFactory.getUTC()
    initialDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
    gmstSeconds = initialDate.getComponents(TimeScalesFactory.getGMST(IERSConventions.IERS_2010, False)).getTime().getSecondsInLocalDay()
    t_sid = (gmstSeconds / 240.0)

    initial_inc, lv, satellite_mass = radians(98.2), radians(0), playerOne[0]
    
    sigma_rp, sigma_ra = 10.0, 10.0
    sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
    sigma_k, sigma_w, sigma_i, sigma_v = 0.01, radians(0.5), radians(0.01), radians(0.5)
    
    mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS, rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
    covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
    sig_ae, gaussian_ae = sigma_points(mean, covariance)

    chunk_results = []
    raan = radians(t_sid)

    for omega in omega_chunk:
        try:
            new_mean = np.zeros((7))
            new_mean[:2] = gaussian_ae.mean()
            new_mean[2], new_mean[3], new_mean[4], new_mean[5], new_mean[6] = 0.0, 1.0, radians(omega), lv, initial_inc
            
            new_covariance = np.zeros((7,7))
            new_covariance[:2, :2] = gaussian_ae.covariance()
            new_covariance[2, 2], new_covariance[3, 3], new_covariance[4, 4] = sigma_t**2, sigma_k**2, sigma_w**2
            new_covariance[5, 5], new_covariance[6, 6] = sigma_v**2, sigma_i**2
            
            sig_points = Gaussian(new_mean, new_covariance).compute_sigma_points()
            pa, inc, LAN = [], [], []
            too_long = False
            for sp in sig_points:

                if too_long == True:
                    random_reentry = random_risk(2020, degrees(initial_inc), playerOne[1])
                    risk = random_reentry/random_reentry
                    chunk_results.append({
                'rp': rp, 'ra': ra, 'perigee_argument': omega,
                'min_casualty_risk': risk, 'min_lan_coordinate': 0,
                'mean_lan': 0, 'mean_perig': 0,
                'cov_00': 0, 'cov_01': 0,
                'cov_10': 0, 'cov_11': 0
            })
                else:
                    # Call the global standalone function. Receives ONLY floats back.
                    pa_val, RAAN_val, incli_val, elapsed = numerical_propagation(
                        sp, raan, initial_inc, radians(omega), lv, initialDate, duration, satellite_mass, playerOne[1], playerOne[2]
                    )
                    
                    if elapsed >= duration:
                        print(f"Warning: Reentry exceeded duration for omega {omega}")
                        too_long = True                        

                    pa.append(pa_val)
                    sid_time_diff = (elapsed/3600)*(360/24) + gmstSeconds/240
                    LAN.append(np.radians(np.degrees(RAAN_val) - sid_time_diff))
                    inc.append(np.degrees(incli_val))
                    #print(f'The inclination for this point is: {np.degrees(incli_val)}')
            print(f'Propagation Complete!')
            if too_long == False:
                unwrapped_pa, lan_unwrapped, sort_idx = unwrap_angles(pa, LAN)
                final_mean, final_covariance = final_mean_covariance(unwrapped_pa, lan_unwrapped, sort_idx, 7)


                popt_i_unwrapped, _ = curve_fit(inclination_model, unwrapped_pa, inc)
                fitted_inc = radians(popt_i_unwrapped[0])
                fitted_inc = radians(97.95)
                print(f'The mean inclination curve fit found was: {fitted_inc}')
                casualty_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(playerOne[1]))**2
                integration_results = jl.compute_casualty_risk(static_cells, final_mean, final_covariance, fitted_inc, casualty_area, 1e-3)
                risk_matrix = np.array(integration_results).reshape((180, 360))

                random_reentry = random_risk(2020, degrees(fitted_inc), playerOne[1])
                
                min_risk_value, min_lan_shift = float('inf'), -1
                risk_list = []
                for shift in range(360):
                    shifted_matrix = np.roll(risk_matrix, shift, axis=1)
                    total = sum(pop * shifted_matrix[r, c] for r, c, pop in pop_targets)
                    normalized_risk = (total * casualty_area) / random_reentry
                    risk_list.append(normalized_risk)
                    
                    if normalized_risk < min_risk_value:
                        min_risk_value = normalized_risk
                        min_lan_shift = shift

                chunk_results.append({
                    'rp': rp, 'ra': ra, 'perigee_argument': omega,
                    'min_casualty_risk': min_risk_value, 'min_lan_coordinate': min_lan_shift,
                    'mean_lan': final_mean[0], 'mean_perig': final_mean[1],
                    'cov_00': final_covariance[0, 0], 'cov_01': final_covariance[0, 1],
                    'cov_10': final_covariance[1, 0], 'cov_11': final_covariance[1, 1]
                })

        except BaseException as e:
            print(f"\n--- WORKER EXCEPTION REVEALED FOR OMEGA {omega} ---\n{traceback.format_exc()}\n---------------------------------------------")
            continue 
            
    return chunk_results

# -------------------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    start_time = time.time()
    pop_targets = load_population_targets("POP_100_2024_GPW_V4_UN_ext.asc")
    
    cell_longitudes, cell_latitudes = np.arange(-180.0, 180.0, 1.0), np.arange(-90.0, 90.0, 1.0)
    static_cells = [(radians(lat), radians(lat + 1.0), radians(lon), radians(lon + 1.0)) 
                    for lat in cell_latitudes for lon in cell_longitudes]

#INPUT THE RANGE OF VALUES OF GRID SEARCHING HERE. IF A SLURM IS TO BE USED, CHANGE THE VALUES INSIDE HERE INTO THE SLURM ARGUMENTS.
#TO CHANGE THE VALUES OF THE UNCERTAINTIES, SEE LINES 232 TO 234.
    rp_values = range(130000, 135000, 10000)
    ra_values = range(270000, 280000, 10000)
    parameter_sets = [{'rp': x, 'ra': y} for x in rp_values for y in ra_values]

#THE NUMBERS OF CORES TO USE CAN BE CHANGED HERE.
    playerOne, duration, num_cores = [90, 1.1, 2.25], 4*24*3600, 3
    wide_format_results = []
    
    for idx, params in enumerate(parameter_sets):
        rp, ra = params['rp'], params['ra']
        print(f"\nAnalyzing Set {idx+1}/{len(parameter_sets)}: Rp = {rp}, Ra = {ra}")

        omega_array = np.arange(270, 450, 1)
        chunks = np.array_split(omega_array, num_cores)

        mp_args = [(chunk, rp, ra, pop_targets, static_cells, duration, playerOne) for chunk in chunks]
        print(f'Starting the multiprocessing with {num_cores} processes')
        with mp.Pool(processes=num_cores) as pool:
            overall_results = pool.starmap(process_omega_chunk, mp_args)

        # Flatten the worker outputs for this specific altitude
        flat_results = [item for sublist in overall_results for item in sublist]
        
        # Guard clause in case a specific altitude fails entirely
        if not flat_results:
            continue

        # Locate the optimal perigee argument (global minimum for this altitude)
        optimal_result = min(flat_results, key=lambda x: x['min_casualty_risk'])
        # Build the wide row template with the optimal covariance and mean
        row_data = {
            'run_id': idx,
            'rp': rp,
            'ra': ra,
            'final_mean': f"[{optimal_result['mean_lan']}  {optimal_result['mean_perig']}]",
            'final_cov_00': optimal_result['cov_00'],
            'final_cov_01': optimal_result['cov_01'],
            'final_cov_10': optimal_result['cov_10'],
            'final_cov_11': optimal_result['cov_11']
        }
        # Populate the 360 columns mapping the perigee argument to its minimum casualty risk
        for res in flat_results:
            omega_val = int(res['perigee_argument'])
            row_data[f'risk_{omega_val}_degrees'] = res['min_casualty_risk']
            row_data[f'{omega_val}_degrees_opt_LAN'] = res['min_lan_coordinate']

        # Append the final descriptors
        row_data['min_value'] = optimal_result['min_casualty_risk']
        row_data['min_perigee_degree'] = int(optimal_result['perigee_argument'])
        row_data['optimal_lan_shift'] = optimal_result['min_lan_coordinate']

        wide_format_results.append(row_data)
    import pandas as pd
    df = pd.DataFrame(wide_format_results)
    df.to_csv('optimisation_results_perigee.csv', index=False)
    print(f"\nExecution Time: {time.time() - start_time:.2f} seconds")
