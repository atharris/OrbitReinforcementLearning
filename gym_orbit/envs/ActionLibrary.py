#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate a mars entry scenario using linear state and estimation error growth models.
"""

# core modules
import random
import math
import copy

# 3rd party modules
import gym
import orbitalMotion as om
import StateLibrary as StateLib
import RigidBodyKinematics as rbk
import numpy as np
# from scipy.integrate import RK45



class mode_options:
    '''
    Class to hold all parameters necessary to propagate the spacecraft state forward through a mode.
    '''
    def __init__(self):
        self.mode_length = 0 #  Total durtion of a mode, s
        self.dt = 0 #   Timestep to be used in a mode, sl
        self.mu = 0 #   Grav parameter for the planet
        self.ref_coeff = 0 #    Reflectivity coefficient for the spacecraft
        self.j2 = 0 #   J2 coeff for the planet
        self.acc = np.zeros([3,]) # Additional acceleration, due to control/perturbation
        self.rp = 0 #   Planet radius
        self.cov_noise = 0.001*np.identity(6)
        self.error_stm = -0.001*np.identity(6)
        self.obs_limit = 1.0 #  Converged estimator accuracy in meters
        self.burn_number = 1 # Allow one DV burn by default
        self.goal_orbel = om.ClassicElements()

def propModel(t,y,odeOptions):
    mu = odeOptions.mu
    acc = odeOptions.acc
    radius = np.linalg.norm(y[0:3])
    gravAcc = -mu/(radius**3.0)
    ctStm = np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[gravAcc,0,0,0,0,0],[0,gravAcc,0,0,0,0],[0,0,gravAcc,0,0,0]])
    ctStm = np.reshape(ctStm, [6,6])
    addAcc = np.array([[0],[0],[0],[acc[0]],[acc[1]],[acc[2]]])
    addAcc = np.reshape(addAcc, [6,])
    y_dot = np.dot(ctStm, y) + addAcc
    y_dot = np.reshape(y_dot, [6,])
    return y_dot

def rk4(fun,t0,dt,y0, funOptions):
    #INPUTS:
    # fun: function to be integrated, defined as ynew = fun(t,y)
    # t0: time at y0
    # dt: designated step size (also ref'd as 'h')
    # y0: initial conditions
    k1 = fun(t0,y0,funOptions)
    k2 = fun(  t0+dt/2.0,  y0 + dt/2.0 * k1,funOptions)
    k3 = fun(t0+dt/2.0, y0 + dt/2.0 * k2,funOptions)
    k4 = fun(t0 + dt, y0 + dt*k3,funOptions)
    y1 = y0 + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

    return y1

def j2PropModel(t,y,odeOptions):
    mu = odeOptions.mu
    acc = odeOptions.acc
    j2 = odeOptions.j2
    planetRad = odeOptions.rp
    radius = np.linalg.norm(y[0:3])

    gravDir = y[0:3] / radius

    y_dot = np.zeros([6,])
    a = np.zeros([3,])
    a[0] = - 3*j2*mu*planetRad**2*y[0]/(2*radius**5)*(1.-5.*y[2]**2/radius**2)
    a[1] = - 3*j2*mu*planetRad**2*y[1]/(2*radius**5)*(1.-5.*y[2]**2/radius**2)
    a[2] = - 3*j2*mu*planetRad**2*y[2]/(2*radius**5)*(3.-5.*y[2]**2/radius**2)
    y_dot[0:3] = y[3:]
    y_dot[3:] = -mu/(radius**2.0) * gravDir + a + acc

    return y_dot

def truth_propagate(input_state,  mode_options):

    init_vec = input_state.state_vec
    #integrator = RK45(fun=lambda t, y: j2PropModel(t, y, mode_options), t0=0, y0=init_vec,  t_bound=mode_options.dt)
    #integrator.step()
    #input_state.state_vec = integrator.y
    input_state.state_vec = rk4(j2PropModel, 0, mode_options.dt, init_vec, mode_options)

    return input_state

def sc_propagate(input_state,  mode_options):

    init_vec = input_state.state_vec
    #integrator = RK45(fun=lambda t, y: propModel(t, y, mode_options), t0=0, y0=init_vec,  t_bound=mode_options.dt)
    #integrator.step()
    #input_state.state_vec = integrator.y
    input_state.state_vec = rk4(propModel, 0, mode_options.dt, init_vec, mode_options)

    return input_state

def sc_htransfer_propagate(input_state,  mode_options):

    init_vec = input_state.state_vec
    elems = om.rv2elem(om.MU_MARS, init_vec[:3], init_vec[3:6])

    if elems.f < 1E-3 and input_state.burns < mode_options.burn_number:
        input_state, DV = resultOrbit(input_state, mode_options.goal_orbel)
        input_state.burns += 1

    input_state.state_vec = rk4(propModel, 0, mode_options.dt, init_vec, mode_options)

    return input_state

def est_propagate(estimated_state, mode_options):
    init_mean = estimated_state.state_vec
    init_cov = estimated_state.covariance

    estimated_state = sc_propagate(estimated_state, mode_options)
    estimated_state.covariance = init_cov + mode_options.cov_noise * mode_options.dt
    return estimated_state

def lyap_controller(ref_state, sc_state, K1, K2, mode_options):

    tmp_opts = copy.deepcopy(mode_options)
    tmp_opts.acc = np.zeros([3,])
    refAcc = propModel(0, ref_state, tmp_opts)
    scAcc = propModel(0, sc_state, tmp_opts)

    posErr = sc_state[0:3] - ref_state[0:3]
    velErr = sc_state[3:] - ref_state[3:]

    dynDiff = scAcc[3:] - refAcc[3:]

    controlOut = -dynDiff - K1.dot(posErr) - K2.dot(velErr)

    return controlOut

def resultOrbit(input_state, desired_orbit):
    '''
    Mode in which the spacecraft is rendered non-functional and state error increases.
    :param input_state: state of the s/c when thrust is commanded
    :param DV: scalar thrust commanded (positive DV is against the s/c velocity)
    :return: oe : Orbital Elements of the resulting orbit (instantiation of ClassicElements class)

    '''
    assert np.shape(input_state.state_vec)[0] == 6

    r_des, v_des = om.elem2rv_parab(om.MU_MARS, desired_orbit)

    r1 = np.linalg.norm(input_state.state_vec[0:3])
    r2 = np.linalg.norm(r_des)

    DV = np.sqrt(om.MU_MARS/r1)*(np.sqrt(2*r2/(r1+r2))-1.)


    thrust = StateLib.rv_state()
    thrust.state_vec[3:6] = DV*input_state.state_vec[3:6]/np.linalg.norm(input_state.state_vec[3:6])

    input_state.state_vec += thrust.state_vec

    return input_state, DV

def observationMode(est_state, ref_state, true_state, mode_options):
    '''
    Function to simulate a period of orbit determination.
    :param est_state: Estimated state at the beginning of the mode.
    :param ref_state: Internal reference state at the beginning of the mode.
    :param true_state: "Ground truth" state the spacecraft acts on and observes.
    :param des_state: The desired end state of the spacecraft.

    :return est_state: Updated state estimate at the end of the mode.
    :return ref_state: Propagated reference state at the end of the mode.
    :return true_state: propagated truth state at the end of the mode.
    '''

    tvec = np. arange(0,mode_options.mode_length, mode_options.dt)
    est_error = -true_state.state_vec + est_state.state_vec
    for ind in range(1,len(tvec)):
        #   Propagate truth state forward:
        true_state = truth_propagate(true_state, mode_options)
        #   Propagate reference state forward:
        ref_state = sc_propagate(ref_state, mode_options)
        #   Generate measurement of true state:                                                         
        est_error = mode_options.error_stm.dot(est_error) + mode_options.obs_limit * np.random.randn(6)

    est_state.state_vec = true_state.state_vec + est_error
    est_state.covariance = mode_options.obs_limit * np.identity(6)

    return est_state, ref_state, true_state

def controlMode(est_state, ref_state, true_state, mode_options):
    '''
        Function to simulate a period of orbit determination.
        :param est_state: Estimated state at the beginning of the mode.
        :param ref_state: Internal reference state at the beginning of the mode.
        :param true_state: "Ground truth" state the spacecraft acts on and observes.
        :param des_state: The desired end state of the spacecraft.

        :return est_state: Updated state estimate at the end of the mode.
        :return ref_state: Propagated reference state at the end of the mode.
        :return true_state: propagated truth state at the end of the mode.
        '''
    tvec = np.arange(0, mode_options.mode_length, mode_options.dt)
    k1 = 0.01*np.identity(3)
    k2 = 0.1*np.identity(3)

    control_use = 0
    ref_options = copy.deepcopy(mode_options)
    ref_options.acc = np.zeros([3,])

    for ind in range(0, len(tvec)):
        #   Compute control acceleration:
        mode_options.acc = lyap_controller(ref_state.state_vec, est_state.state_vec, k1, k2, mode_options)
        control_use = control_use + abs(np.linalg.norm(mode_options.acc))
        #   Propagate truth state forward:
        true_state = truth_propagate(true_state, mode_options)
        #   Propagate reference state forward:
        ref_state = sc_propagate(ref_state, ref_options)
        est_state = est_propagate(est_state, mode_options)

    mode_options.acc = np.zeros([3,])
    return est_state, ref_state, true_state, control_use

def thrustMode(est_state, ref_state, true_state, mode_options):
    '''
        Function to simulate a period of orbit determination.
        :param est_state: Estimated state at the beginning of the mode.
        :param ref_state: Internal reference state at the beginning of the mode.
        :param true_state: "Ground truth" state the spacecraft acts on and observes.
        :param des_state: The desired end state of the spacecraft.

        :return est_state: Updated state estimate at the end of the mode.
        :return ref_state: Propagated reference state at the end of the mode.
        :return true_state: propagated truth state at the end of the mode.
        '''
    tvec = np.arange(0, mode_options.mode_length, mode_options.dt)

    ref_options = copy.deepcopy(mode_options)
    ref_options.acc = np.zeros([3,])

    #   Compute control acceleration:
    true_state, DVtruth = resultOrbit(est_state, mode_options.goal_orbel)
    est_state, DVest = resultOrbit(est_state, mode_options.goal_orbel)
    control_use = DVest

    for ind in range(0, len(tvec)):
        #   Propagate est/truth state forward after DV
        true_state = truth_propagate(true_state, mode_options)
        est_state = est_propagate(est_state, mode_options)
        #   Propagate reference state forward with transfer dynamics
        ref_state = sc_htransfer_propagate(ref_state, ref_options)

    mode_options.acc = np.zeros([3,])
    return est_state, ref_state, true_state, control_use

