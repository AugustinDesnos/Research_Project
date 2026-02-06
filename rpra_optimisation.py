# The Gaussian class and unscented_transform used in this code are from:
# https://github.com/hugohadfield/unscented_transform

import orekit
vm = orekit.initVM() # Initialize the JVM once at the top

from math import radians, degrees
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

# Orekit Imports
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
from scipy.interpolate import RectBivariateSpline
import multiprocessing as mp
import numpy as np
from unscented_transform import Gaussian, unscented_transform
from scipy.optimize import curve_fit


# --- HELPER FUNCTIONS ---

def load_population_data(filename):
    """
    Loads the population data once to avoid I/O bottlenecks.
    Returns a list of coordinate tuples with population values.
    """
    print(f"Loading population data from {filename}...")
    try:
        ncols, nrows = 360, 145
        xllcorner, yllcorner = -180.0, -60.0
        cellsize = 1.0
        
        values = []
        with open(filename, "r") as f:
            lines = f.readlines()[6:] # Skip header
            for line in lines:
                values.extend(map(float, line.split()))

        values = np.array(values).reshape((nrows, ncols))

        longitudes = np.arange(xllcorner, xllcorner + ncols * cellsize, cellsize)
        latitudes = np.arange(yllcorner, yllcorner + nrows * cellsize, cellsize)
        latitudes = np.flip(latitudes)

        GRID_SIZE = 1
        # Pre-process into the format Julia expects
        coords_with_values = []
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                val = values[i][j]
                if val > 0: # Optimization: only store populated cells
                    coords_with_values.append((
                        (radians(lat), radians(lat + GRID_SIZE), radians(lon), radians(lon + GRID_SIZE)),
                        val
                    ))
        print("Population data loaded.")
        return coords_with_values
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

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

def unwrap_angles(perigee, lan):
    unwrapped = [perigee[0]]
    old_perigee = perigee[3:6]
    old_lan = lan[3:6]
    index = [3, 4, 5]
    perigee = np.delete(perigee, index)
    lan = np.delete(lan, index)
    if perigee[2] > perigee[1]:
        alpha_grad = (perigee[1] - 2*np.pi - perigee[2]) - perigee[1]
        lan_grad = lan[2] - lan[1]
    else:
        alpha_grad = perigee[2] - perigee[1]
        lan_grad = lan[2] - lan[1]
    lan_period = lan_grad*2*np.pi/(-alpha_grad)
    for i in range(1, len(perigee)):
        if lan[i] - lan[i-1] < lan_period:
            if perigee[i] > perigee[i-1]:
                unwrapped.append(unwrapped[-1] + (perigee[i] - 2*np.pi - perigee[i-1]))
            else:
                unwrapped.append(unwrapped[-1] + (perigee[i] - perigee[i-1]))
        else:
            x = np.floor((lan[i]-lan[i-1])/lan_period)
            unwrapped.append(unwrapped[-1] + (perigee[i] - 2*x*np.pi - perigee[i-1]))
    alpha_grad = unwrapped[-1] - unwrapped[0]
    lan_grad = lan[-1] - lan[0]
    lan = np.append(lan, old_lan[1])
    lan = np.append(lan, old_lan[0])
    lan = np.append(lan, old_lan[2])
    unwrapped.append(unwrapped[1] + (old_lan[1] - lan[1])*alpha_grad/lan_grad)
    unwrapped.append(old_perigee[0] - (old_perigee[1] - unwrapped[-1]))
    unwrapped.append(old_perigee[2] - (old_perigee[1] - unwrapped[-2]))
    sort_index = np.argsort(lan)
    unwrapped = np.array(unwrapped)
    unwrapped = unwrapped[sort_index]
    return unwrapped

def inclination_model(alpha, i0, i1, i2):
    return i0 + i1 * np.cos(alpha) + i2 * np.cos(2 * alpha)

#code to determine the size of the casualty area required for a random reentry to have a casualty risk of
#1e-4. The data set used was taken from CNES: guide_des_bonnes_pratiques_los_satellites_-_draft.pdf
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
    sat_cas_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(spacecraft_area))**2

    return (1e-4)*sat_cas_area/cas_area


# --- CORE PROPAGATION FUNCTION ---

def numerical_propagation(x, raan, i, omega, lv, initial_date, duration, mass, drag_area, drag_coeff):
    """
    Propagates a single sigma point.
    Arguments must be passed explicitly to support multiprocessing.
    """
    # Re-initialize frames and bodies locally to ensure thread safety in the worker
    inertialFrame = FramesFactory.getEME2000()
    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                             Constants.WGS84_EARTH_FLATTENING, ITRF)
    
    # x contains: [a, e, time_shift, drag_density_factor]
    
    current_date = initial_date.shiftedBy(float(x[2]))
    
    initialOrbit = KeplerianOrbit(float(x[0]), float(x[1]),
                                  float(i), float(omega), float(raan), float(lv),
                                  PositionAngleType.TRUE,
                                  inertialFrame, current_date, Constants.WGS84_EARTH_MU)

    # Integrator Setup
    minStep = 0.0001
    maxstep = 1000.0
    initStep = 10.0
    positionTolerance = 0.01
    tolerances = NumericalPropagator.tolerances(positionTolerance, initialOrbit, initialOrbit.getType())
    integrator = DormandPrince853Integrator(minStep, maxstep, 
        JArray_double.cast_(tolerances[0]), JArray_double.cast_(tolerances[1]))
    integrator.setInitialStepSize(initStep)

    # Propagator Setup
    initialState = SpacecraftState(initialOrbit, float(mass)) 
    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(OrbitType.CARTESIAN)
    propagator_num.setInitialState(initialState)

    # Events and Forces
    altitude_detector = AltitudeDetector(80000.0, earth).withHandler(StopOnEvent())
    propagator_num.addEventDetector(altitude_detector)
    
    sun = CelestialBodyFactory.getSun()
    msafe = MarshallSolarActivityFutureEstimation(
        MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
        MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)
    atmosphere = NRLMSISE00(msafe, sun, earth)
    
    # x[3] is the sigma multiplier for density
    drag_model = IsotropicDrag(drag_area, float(x[3] * drag_coeff))
    drag_force = DragForce(atmosphere, drag_model)
    propagator_num.addForceModel(drag_force)

    gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(ITRF, gravityProvider))
    
    finalDate = initial_date.shiftedBy(float(duration))
    
    # Run Propagation
    final_state = propagator_num.propagate(initial_date, finalDate)
    
    pv = final_state.getPVCoordinates()
    orbit = final_state.getOrbit()
    # We return the SpacecraftState and Orbit. Note: These are Java objects.
    # We must extract data from them immediately in the caller (process_lan)
    return pv, final_state, orbit


# --- WORKER FUNCTION ---

def process_lan(lan_chunk, sig_points, a, e, i, omega, lv, coords_data, duration, mass, drag_area, drag_coeff, epoch_date_str):
    """
    Worker function to process a chunk of LAN values.
    """
    from juliacall import Main as jl
    from juliacall import Pkg as jlPkg
    jlPkg.activate("CasualtyRisk.jl") 
    jlPkg.instantiate()
    jl.seval('push!(LOAD_PATH, pwd())')
    jl.seval("using CasualtyRisk")

    # Attach this thread to the JVM
    if not orekit.getVMEnv():
        vm.attachCurrentThread()
        
    utc = TimeScalesFactory.getUTC()
    epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
    initialDate = epochDate
    
    gmst = TimeScalesFactory.getGMST(IERSConventions.IERS_2010, False)
    gmstSeconds = initialDate.getComponents(gmst).getTime().getSecondsInLocalDay()
    t_sid = (gmstSeconds / 240.0) % 360.0
    
    results = []
    
    #print(f"Starting chunk: {lan_chunk}")

    for lan_value in lan_chunk:
        casualty_risk = [] # Initialize list
        raan = radians(lan_value + t_sid)
        
        pa = []
        inc = []
        LAN = []
        
        # Propagate all sigma points for this specific LAN
        for j in range(len(sig_points)):
            try:
                # Pass all arguments explicitly
                pvs, final_state, raw_orbit = numerical_propagation(
                    sig_points[j], raan, i, omega, lv, 
                    initialDate, duration, mass, drag_area, drag_coeff
                )
                final_state_orbit = KeplerianOrbit(raw_orbit)
                
                # Check duration
                elapsed = final_state.getDate().durationFrom(initialDate)
                if abs(elapsed - duration) < 1.0: # If it didn't crash early (didn't re-enter)
                     print("Stopping: reentry takes longer than max duration")
                     # Treat as 0 risk or handle accordingly
                     casualty_risk.append(random_risk(2020, 97.2, 1.1))
                     break
                
                # Data Extraction
                RAAN_val = final_state_orbit.getRightAscensionOfAscendingNode()
                perig = final_state_orbit.getPerigeeArgument()
                incli = final_state_orbit.getI()
                trueano = final_state_orbit.getTrueAnomaly()
                
                pa.append(perig + trueano)
                
                gmstSeconds_final = final_state.getDate().getComponents(gmst).getTime().getSecondsInLocalDay()
                sid_time_final = (gmstSeconds_final / 240.0) % 360.0
                
                LAN.append(np.radians(np.degrees(RAAN_val) - sid_time_final))
                inc.append(np.degrees(incli))
                
            except Exception as e:
                print(f"Propagation error at LAN {lan_value}, point {j}: {e}")
                continue

        # If loop finished successfully and we have data
        if len(LAN) == len(sig_points):
            LAN = np.array(LAN)
            pa = np.array(pa)
            inc = np.array(inc)
            
            # Sort
            sorted_indices = np.argsort(LAN)
            LAN_sorted = LAN[sorted_indices]
            pa_sorted = pa[sorted_indices]
            inc_sorted = inc[sorted_indices]
            
            # Unwrap
            unwrapped_pa = np.array(unwrap_angles(pa_sorted, LAN_sorted))
            
            # Match original sigma indices to sorted indices
            sigma_indices = list(range(len(sig_points)))
            reverse_lookup = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}
            sigma_sorted_indices = [reverse_lookup[k] for k in sigma_indices]

            # Reconstruct Sigma Points in 2D (LAN, Alpha)
            # zip LAN and unwrapped_pa based on sigma sorted order
            final_sig_points_list = []
            for k in sigma_sorted_indices:
                final_sig_points_list.append([LAN_sorted[k], unwrapped_pa[k]])
            
            final_sig_points = np.array(final_sig_points_list)
            
            # Calculate Mean and Covariance
            final_mean = np.mean(final_sig_points, axis=0)
            final_covariance = np.cov(final_sig_points, rowvar=False)
            
            # Curve fit for inclination
            try:
                popt_i_unwrapped, _ = curve_fit(inclination_model, unwrapped_pa[sigma_sorted_indices], inc[:len(sig_points)])
                inc_val = radians(popt_i_unwrapped[0]) # Use the mean term (i0)
            except:
                inc_val = radians(np.mean(inc))

            # Julia Calculation
            sat_cas_area = (np.sqrt((0.677/2)**2*np.pi) + np.sqrt(drag_area))**2
            
            # Using the pre-loaded coords_data
            try:
                total_result = jl.compute_casualty_risk(coords_data, final_mean, final_covariance, inc_val, sat_cas_area, 1e-3)
                casualty_risk.append(total_result)
                #print(f"LAN {lan_value}: Risk = {total_result}")
            except Exception as e:
                print(f"Julia error: {e}")
                casualty_risk.append(0)
        random_reentry = random_risk(2020, 97.2, 1.1)
        if casualty_risk:
            # Fix: Append tuple
            results.append((min(casualty_risk/random_reentry), lan_value))

    return results


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("===================================================================================================================")
    print("Initializing parameters...")
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # 1. Load Population Data ONCE
    population_file = "POP_100_2024_GPW_V4_UN_ext.asc"
    coords_with_values = load_population_data(population_file)
    
    if not coords_with_values:
        print("CRITICAL: Population data not loaded. Check file path.")
        exit()

    # 2. Setup Time and Frames
    utc = TimeScalesFactory.getUTC()
    epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
    
    # 3. Define Grid
    rp_grid = np.arange(130000.0, 170000.0, 1000.0)
    ra_grid = np.arange(305000.0, 310000.0, 1000.0)
    RP, RA = np.meshgrid(rp_grid, ra_grid)
    min_cas = []
    # 4. Loop through Grids
    for rp, ra in zip(RP.ravel(), RA.ravel()):
        
        #print(f"\nAnalyzing: Rp = {rp}, Ra = {ra}")
        
        # Satellite Parameters
        i = radians(98.2)
        omega = radians(0)
        lv = radians(0)
        satellite_mass = 90.0
        playerOne = [90, 1.1, 2.25] # Mass, Area, DragCoeff
        
        a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0   
        e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a
        
        # Unscented Transform Setup
        duration = 4*24*3600
        sigma_rp = 500.0
        sigma_ra = 1000.0
        sigma_t = np.radians(0.5)/Constants.WGS84_EARTH_ANGULAR_VELOCITY
        sigma_k = 0.1
        
        mean = np.array([ra+Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                         rp+Constants.WGS84_EARTH_EQUATORIAL_RADIUS])
        covariance = np.array([[sigma_ra**2, 0.0], [0.0, sigma_rp**2]])
        
        # Transform rp, ra -> a, e
        sig_ae, gaussian_ae = sigma_points(mean, covariance)
        
        # Create full gaussian (a, e, t, k)
        new_mean = np.zeros((4))
        new_mean[:2] = gaussian_ae.mean()
        new_mean[2] = 0.0
        new_mean[3] = 1.0
        
        new_covariance = np.zeros((4,4))
        new_covariance[:2, :2] = gaussian_ae.covariance()
        new_covariance[2, 2] = sigma_t**2
        new_covariance[3, 3] = sigma_k**2
        
        newgaussian = Gaussian(new_mean, new_covariance)
        sig_points = newgaussian.compute_sigma_points()

        # Multiprocessing Setup
        lans = np.arange(0, 360, 1)
        n_chunks = mp.cpu_count()
        chunks = np.array_split(lans, n_chunks)
        
        # Arguments for workers
        # passing 'epochDate' might fail if pickled, so we reconstruct it in worker
        # We pass necessary simple types and the large data array
        mp_args = [(chunk, sig_points, a, e, i, omega, lv, coords_with_values, 
                    duration, playerOne[0], playerOne[1], playerOne[2], "2020-01-01") 
                   for chunk in chunks]

        print(f"Launching {n_chunks} worker processes...")
        
        with mp.Pool() as pool:
            overall_results = pool.starmap(process_lan, mp_args)
        # Flatten results
        flat_results = [item for sublist in overall_results for item in sublist]
        
        if flat_results:
            cas_results, lan_values = zip(*flat_results)
        else:
            print("No valid results computed.")
        min_cas.append(min(flat_results, key=lambda x: x[0]))
        print(min_cas)
    results_array = np.array([min_cas[i][0] for i in range(len(min_cas))])
    Z_matrix = results_array.reshape(RP.shape)
    print(min_cas)
    print(rp_grid)
    print(ra_grid)