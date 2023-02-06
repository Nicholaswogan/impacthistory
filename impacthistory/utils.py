from scipy import interpolate
import numpy as np
import numba as nb

from . import constants as const

# Model for lunar cratering record

@nb.njit()
def num_lunar_crater_bigger_1km_per_area(t, a, b, c):
    return 10.0**(a*np.exp(b*t) + c)

##################
### Custom SFD ###
##################

# SFD of main belt asteroids

def main_belt_SFD_scaled_to_1km_lunar():
    """Scale the main belt SFD to a asteroid that creates a 
    1 km crater on the moon. I follow Morbidelli et al (2018), 
    Sec. 2.2, assuming the projectile has certain density and 
    diameter."""
    rho_p = 2.5e12 # kg/km^3
    D_projectile = 50e-3 # km
    m0 = diameter_to_mass(D_projectile, rho_p)
    SFD_masses, SFD_frequency = main_belt_SFD_scaled_to_mass(m0)
    return SFD_masses, SFD_frequency

def main_belt_SFD_scaled_to_mass(m0):
    # Table 1, Morbidelli et al. (2018)
    D = np.array([1, 10, 12, 18, 75, 900])
    S = np.array([1100000,8000,5400,2500,370,1])
    
    # Convert to impact mass
    rho_p = 2.5e12 # kg/km^3 from Morbidelli et al. (2018)
    M = diameter_to_mass(D, rho_p)
    
    # Extrapolate SFD down to 1 kg
    a = interpolate.interp1d(np.log10(M), np.log10(S),fill_value='extrapolate')
    S_1kg = 10.0**a(np.log10(1.0)).item()
    M = np.append(1.0,M)
    S = np.append(S_1kg,S)
    
    # Rescale SFD so that it is equal to 1 at m0
    a = interpolate.interp1d(np.log10(M), np.log10(S),fill_value='extrapolate')    
    S0 = 10.0**a(np.log10(m0)).item()
    S = S/S0
    
    return M, S

@nb.njit()
def extend_SFD(M, S, b, M_max):

    if M_max < M[-1]:
        return M, S

    S_max = (S[-1]/M[-1]**-b)*M_max**-b
    
    M_new = np.append(M, M_max)
    S_new = np.append(S, S_max)
    
    return M_new, S_new

# Number of impactors

@nb.njit()
def num_impactors_per_area_moon_custom_SFD(m, log10_M, log10_S, t2, t1, a, b, c):
    log10_S_at_m = np.interp(np.log10(m),log10_M,log10_S)
    S_at_m = 10.0**log10_S_at_m
    num = (10.0**(a*np.exp(b*t2) + c) - 10.0**(a*np.exp(b*t1) + c))*S_at_m
    return num

@nb.njit()
def num_impactors_moon_custom_SFD(m, log10_M, log10_S, t2, t1, a, b, c):
    return const.area_moon*num_impactors_per_area_moon_custom_SFD(m, log10_M, log10_S, t2, t1, a, b, c)

@nb.njit()
def num_impactors_earth_custom_SFD(m, log10_M, log10_S, t2, t1, a, b, c, v_inf):
    factor = earth_moon_gravitational_scaling_factor(v_inf)
    return const.area_earth*factor*num_impactors_per_area_moon_custom_SFD(m, log10_M, log10_S, t2, t1, a, b, c)

# gravitational scaling

@nb.njit()
def gravitational_scaling_factor(vesc_1, vesc_2, v_inf):
    a1 = (1+(vesc_1/v_inf)**2)
    a2 = (1+(vesc_2/v_inf)**2)
    return a1/a2

@nb.njit()
def earth_moon_gravitational_scaling_factor(v_inf):
    return gravitational_scaling_factor(const.escape_velocity_earth, const.escape_velocity_moon, v_inf)

# misc

@nb.njit()
def mass_to_diameter(m, rho): # in kg
    # rho in kg/km3
    return 2*((3/(4*np.pi))*m/rho)**(1/3) # km

@nb.njit()
def diameter_to_mass(D, rho): # km
    # rho in kg/km3
    return rho*(4/3)*np.pi*(D/2)**3 # mass in kg
