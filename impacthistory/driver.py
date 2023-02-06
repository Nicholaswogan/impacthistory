import numpy as np
from matplotlib import pyplot as plt
from numba import types
import numba as nb
import os

from . import utils

data_dir = os.path.dirname(os.path.realpath(__file__))+'/data/'

def create_impact_velocity_sampler():

    v, P = np.loadtxt(data_dir+'velocity_distribution.txt').T
    P = np.clip(P, a_min=0.0, a_max = np.inf)
    
    v_bins = np.linspace(0,70,1000)
    v_center = (v_bins[1:] + v_bins[:-1])/2
    P_new = np.interp(v_center, v, P)
    
    hist_cumulative = np.cumsum(P_new / P_new.sum())
    
    @nb.njit()
    def sampler(size = 1):
        return np.interp(np.random.uniform(0, 1, size = size), hist_cumulative, v_center)

    return sampler

def create_impact_angle_sampler():
    
    @nb.njit()
    def impact_angle_inverse_cdf(P):
        return (180/np.pi)*0.5*np.arccos(-2*P + 1)
    
    @nb.njit()
    def sampler(size = 1):
        return impact_angle_inverse_cdf(np.random.uniform(0, 1, size = size))

    return sampler

impact_velocity_sampler = create_impact_velocity_sampler()
impact_angle_sampler = create_impact_angle_sampler()

@nb.experimental.jitclass()
class AnglesVelocities():
    mass_min : types.double
    ind_mass_min : types.int32
    inds_start : types.uint64[:,:,:]
    inds_len : types.uint32[:,:,:]
    angles : types.double[:]
    velocities : types.double[:]

    def __init__(self, p, N, mass_min):
        
        self.mass_min = mass_min
        self.ind_mass_min = p.ind_of_bigger_mass(mass_min)
        
        self.inds_start = \
            np.empty((N.shape[0],N.shape[1],N.shape[2]-self.ind_mass_min),np.int64)
        self.inds_len = \
            np.empty((N.shape[0],N.shape[1],N.shape[2]-self.ind_mass_min),np.int32)

        self.angles = np.empty((0),np.float64)
        self.velocities = np.empty((0),np.float64)
        ind = 0
        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                for k in range(self.ind_mass_min, N.shape[2]):
                    kk = k - self.ind_mass_min
                    
                    if N[i,j,k] > 0:
                        angles = impact_angle_sampler(N[i,j,k])
                        velocities = impact_velocity_sampler(N[i,j,k])

                        self.angles = np.append(self.angles, angles)
                        self.velocities = np.append(self.velocities, velocities)

                        self.inds_start[i,j,kk] = ind
                        self.inds_len[i,j,kk] = len(angles)

                        ind += len(angles)
                    else:
                        self.inds_start[i,j,kk] = ind
                        self.inds_len[i,j,kk] = 0

@nb.experimental.jitclass()
class ImpactMonteCarlo():
    a : types.double
    b : types.double
    c : types.double
    tau : types.double
    SFD_masses : types.double[:]
    SFD_frequency : types.double[:]
    mass_grid : types.double[:]
    time_grid : types.double[:]
    v_inf : types.double
    b_low : types.double
    b_high : types.double

    mass_grid_avg : types.double[:]
    time_grid_avg : types.double[:]
    timeout_iters : types.int32
    
    def __init__(self, a, b, c, SFD_masses, SFD_frequency,
                 mass_grid, time_grid, v_inf, b_low, b_high):
        self.a = a
        self.b = b
        self.c = c
        self.SFD_masses = SFD_masses
        self.SFD_frequency = SFD_frequency
        self.mass_grid = mass_grid
        self.time_grid = time_grid
        self.v_inf = v_inf
        self.b_low = b_low
        self.b_high = b_high
        
        self.mass_grid_avg = 0.5*(mass_grid[1:] + mass_grid[:-1])
        self.time_grid_avg = 0.5*(time_grid[1:] + time_grid[:-1])
        self.timeout_iters = 1000
        
    def sample_impact_history(self, N, b_extend, M):
        
        b_extend[()] = np.random.uniform(self.b_low, self.b_high)

        M_extend, S_extend = utils.extend_SFD(self.SFD_masses, self.SFD_frequency, b_extend[()], self.mass_grid[-1]*2)
        log10_M_extend = np.log10(M_extend)
        log10_S_extend = np.log10(S_extend)
        
        M[()] = 0.0
        for i in range(self.time_grid.shape[0] - 1):
            for j in range(self.mass_grid.shape[0] - 1):
                N1 = utils.num_impactors_earth_custom_SFD(
                    self.mass_grid[j], log10_M_extend, log10_S_extend, 
                    self.time_grid[i], self.time_grid[i+1], self.a, self.b, self.c, self.v_inf
                )
                N2 = utils.num_impactors_earth_custom_SFD(
                    self.mass_grid[j+1], log10_M_extend, log10_S_extend, 
                    self.time_grid[i], self.time_grid[i+1], self.a, self.b, self.c, self.v_inf
                )
                N_average = N1 - N2
                N[i,j] = np.random.poisson(N_average)
                
                M[()] += N[i,j]*self.mass_grid_avg[j]
                
    def sample_impact_history_mass_constraint(self, N, b_extend, M, M_low, M_high):
        
        i = 0
        while True:
            self.sample_impact_history(N, b_extend, M)
            i += 1
            if M[()] > M_low and M[()] < M_high:
                break
            if i > self.timeout_iters:
                raise Exception('Failed to find impact history that matches mass constraints')
                
    def impact_history(self, niters):

        N = np.zeros((niters,self.time_grid.shape[0] - 1,self.mass_grid.shape[0] - 1), np.int32)
        b_extend = np.zeros((niters,))
        M = np.zeros((niters,))
        b_tmp = np.array(0.0)
        M_tmp = np.array(0.0)

        for i in range(niters):
            self.sample_impact_history(N[i,:,:], b_tmp, M_tmp)
            b_extend[i] = b_tmp[()]
            M[i] = M_tmp[()]

        return N, b_extend, M
    
    def impact_history_mass_constraint(self, niters, M_low, M_high):

        N = np.zeros((niters,self.time_grid.shape[0] - 1,self.mass_grid.shape[0] - 1), np.int32)
        b_extend = np.zeros((niters,))
        M = np.zeros((niters,))
        b_tmp = np.array(0.0)
        M_tmp = np.array(0.0)

        for i in range(niters):
            self.sample_impact_history_mass_constraint(N[i,:,:], b_tmp, M_tmp, M_low, M_high)
            b_extend[i] = b_tmp[()]
            M[i] = M_tmp[()]

        return N, b_extend, M
    
    def ind_of_bigger_mass(self, mass):
        
        ind = -1
        for i in range(self.mass_grid_avg.shape[0]):
            if self.mass_grid_avg[i] > mass:
                ind = i
                break
                
        if ind == -1:
            raise Exception('mass is not in the mass grid')
            
        return ind

    def assign_angles_and_velocities(self, N, mass_min):
        return AnglesVelocities(self, N, mass_min)

    def number_of_impacts_mass_range(self, N, mass_rng):
        
        num = np.zeros(N.shape[0],np.int32)
                
        for i in range(N.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(self.mass_grid.shape[0] - 1):
                    if mass_rng[0] < self.mass_grid_avg[k] < mass_rng[1]:
                        num[i] += N[i,j,k]
                    
        return num
    
    def time_of_last_mass_range(self, N, mass_rng):
        
        time = np.ones(N.shape[0])*4.5
        
        for i in range(N.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(self.mass_grid.shape[0] - 1):
                    if mass_rng[0] < self.mass_grid_avg[k] < mass_rng[1] and \
                       N[i,j,k] > 0 and self.time_grid_avg[j] < time[i]:
                        time[i] = self.time_grid_avg[j]
                    
        return time
    
    def number_of_impacts_in_interval(self, av, mass_rng, theta_rng, v_rng):
        
        if mass_rng[0] < av.mass_min:
            raise Exception('The mass_rng exceeds the mass bounds in sampled angles and velocities')
            
        num = np.zeros(av.inds_start.shape[0],np.int32)
                
        for i in range(av.inds_start.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(self.mass_grid.shape[0] - 1):
                    if mass_rng[0] < self.mass_grid_avg[k] < mass_rng[1]:
                        
                        kk = k - av.ind_mass_min
                        ind1 = av.inds_start[i,j,kk]
                        ind2 = ind1 + av.inds_len[i,j,kk]
                        
                        angles = av.angles[ind1:ind2]
                        v = av.velocities[ind1:ind2]
                        
                        for ii in range(angles.shape[0]):
                            if theta_rng[0] < angles[ii] < theta_rng[1] and v_rng[0] < v[ii] < v_rng[1]:
                                num[i] += 1
                    
        return num

    def time_of_last_in_interval(self, av, mass_rng, theta_rng, v_rng):
        
        if mass_rng[0] < av.mass_min:
            raise Exception('The mass_rng exceeds the mass bounds in sampled angles and velocities')
            
        time = np.ones(av.inds_start.shape[0])*4.5
                
        for i in range(av.inds_start.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(self.mass_grid.shape[0] - 1):
                    if mass_rng[0] < self.mass_grid_avg[k] < mass_rng[1]:
                        
                        kk = k - av.ind_mass_min
                        ind1 = av.inds_start[i,j,kk]
                        ind2 = ind1 + av.inds_len[i,j,kk]
                        
                        angles = av.angles[ind1:ind2]
                        v = av.velocities[ind1:ind2]
                        
                        for ii in range(angles.shape[0]):
                            if theta_rng[0] < angles[ii] < theta_rng[1] and v_rng[0] < v[ii] < v_rng[1]:
                                if self.time_grid_avg[j] < time[i]:
                                    time[i] = self.time_grid_avg[j]
                    
        return time

    def number_of_impacts_in_energy_interval(self, av, energy_rng, theta_rng):
        
        num = np.zeros(av.inds_start.shape[0],np.int32)
                
        for i in range(av.inds_start.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(av.ind_mass_min, self.mass_grid.shape[0] - 1):
                    
                    kk = k - av.ind_mass_min
                    ind1 = av.inds_start[i,j,kk]
                    ind2 = ind1 + av.inds_len[i,j,kk]
                    
                    angles = av.angles[ind1:ind2]
                    v = av.velocities[ind1:ind2]
                    
                    for ii in range(angles.shape[0]):
                        energy = 0.5*self.mass_grid_avg[k]*(v[ii]*1000.0)**2.0

                        if theta_rng[0] < angles[ii] < theta_rng[1] and energy_rng[0] < energy < energy_rng[1]:
                            num[i] += 1
                    
        return num

    def time_of_last_in_energy_interval(self, av, energy_rng, theta_rng):
        
        time = np.ones(av.inds_start.shape[0])*4.5
                
        for i in range(av.inds_start.shape[0]):
            for j in range(self.time_grid.shape[0] - 1):
                for k in range(av.ind_mass_min, self.mass_grid.shape[0] - 1):
                        
                    kk = k - av.ind_mass_min
                    ind1 = av.inds_start[i,j,kk]
                    ind2 = ind1 + av.inds_len[i,j,kk]
                    
                    angles = av.angles[ind1:ind2]
                    v = av.velocities[ind1:ind2]
                    
                    for ii in range(angles.shape[0]):
                        energy = 0.5*self.mass_grid_avg[k]*(v[ii]*1000.0)**2.0

                        if theta_rng[0] < angles[ii] < theta_rng[1] and energy_rng[0] < energy < energy_rng[1]:
                            if self.time_grid_avg[j] < time[i]:
                                time[i] = self.time_grid_avg[j]
                    
        return time