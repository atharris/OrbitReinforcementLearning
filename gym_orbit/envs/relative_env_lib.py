
"""
Simulate a mars entry scenario using nonlinear relative state and estimation error growth models.
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces
import keplerLib as kl
import perturbationLib as pl
import schaubLib as sl
import transferLib as tl

#!/usr/bin/env python
# -*- coding: utf-8 -*-

class relative_spacecraft_state():
    def __init__(self):
        # type: () -> object
        #self.state_err = 0
        #self.est_err= 0
        self.state_vec = np.zeros([6,])
        self.error_mode = 0

class linear_system():
    def __init__(self):
        self.A = np.array([1])
        self.B = np.array([1])
        self.C = np.array([1])
        self.D = np.array([1])

def linear_kalman_step(prior_state, prior_cov, meas, sys, state_noise, meas_noise)
    x_pred = sys.A.dot(prior_state)
    P_pred = sys.A.dot(prior_cov.dot(np.transpose(sys.A))) + state_noise

    covar_surprise = sys.C.dot(P_pred.dot(np.transpose(sys.C))) + meas_noise
    K_new = P_pred.dot(np.transpose(sys.C)).dot(np.linalg.inv(covar_surprise))

    meas_surprise = meas - sys.C.dot(x_pred)
    x_new = x_pred + K_new.dot(meas_surprise)
    P_new = (np.identity(prior_cov.shape) - K_new.dot( H)).dot(P_pred)

    return x_new, P_new

def obs_mode(input_state, time, obs_constants):
    '''
    Propagates the spacecraft state forward. Generates measurements of the relative posititon each minute. Estimates the
    relative state with each new position measurement.
    :param input_state:
    :param time:
    :param obs_constants:
    :return:
    '''
    tvec = np.arange(0,time, obs_constants.dt)
    relpos = np.zeros([6,len(tvec)])
    relpos[:,0] = input_state.state_vec
    est_relpos = input_state.state_est
    est_cov = input_state.cov_est

    for ind in range(1,len(tvec)):
        #   Propagate the true state forward
        relpos[:,ind] = tl.cwPropagate(relpos[:,ind-1],obs_constants.nu, tvec[ind]-tvec[ind-1])

        meas = np.random.multivariate_normal(relpos[0:3,ind], obs_constants.state_noise, 1)

        est_relpos, est_cov = linear_kalman_step(est_relpos, est_cov, meas, obs_constants.sys,
                                                 obs_constants.state_noise, obs_constants.meas_noise)

    output_state = relative_spacecraft_state()
    output_state.state_vec = relpos[:,-1]
    output_state.state_est = est_relpos
    output_state.cov_est = est_cov
    return output_state

def control_mode(input_state, time, control_constants):
    '''
    Propagates the spacecraft state forward. Generates measurements of the relative posititon each minute. Estimates the
    relative state with each new position measurement.
    :param input_state:
    :param time:
    :param obs_constants:
    :return:
    '''
    tvec = np.arange(0,time, obs_constants.dt)
    relpos = np.zeros([6,len(tvec)])
    relpos[:,0] = input_state.state_vec
    est_relpos = input_state.state_est
    est_cov = input_state.cov_est

    for ind in range(1,len(tvec)):
        #   Propagate the true state forward
        relpos[:,ind] = tl.cwPropagate(relpos[:,ind-1],obs_constants.nu, tvec[ind]-tvec[ind-1])

        meas = np.random.multivariate_normal(relpos[0:3,ind], obs_constants.state_noise, 1)

        est_relpos, est_cov = linear_kalman_step(est_relpos, est_cov, meas, obs_constants.sys,
                                                 obs_constants.state_noise, obs_constants.meas_noise)

    output_state = relative_spacecraft_state()
    output_state.state_vec = relpos[:,-1]
    output_state.state_est = est_relpos
    output_state.cov_est = est_cov
    return output_state


def operational_mode(input_state, time, ops_constants):
    '''
    :param input_state:
    :param time:
    :param obs_constants: length-3 list in the following order:
                                    [ obs_estimation_constant, obs_control_constant, obs_error_constant
    :return:
    '''
    out_state = full_spacecraft_state()

    out_state.state_vec = ops_constants[0].dot(input_state.state_vec) * time + input_state.state_vec
    error_draw = np.random.uniform(0,1,[1,])
    if error_draw > ops_constants[1]:
        out_state.error_mode = 1
    else:
        out_state.error_mode = 0

    return out_state

def error_mode(input_state, time, error_constants):
    '''
    Mode in which the spacecraft is rendered non-functional and state error increases.
    :param input_state:
    :param time:
    :param error_constants:
    :return:
    '''
    out_state = spacecraft_state()
    out_state.state_vec = error_constants[0].dot(input_state.state_vec) * time + input_state.state_vec
    out_state.error_mode = 1

    return out_state

def safe_mode(input_state, time, safe_constants):
    '''
    Mode in which the spacecraft is rendered non-functional and state error increases.
    :param input_state:
    :param time:
    :param error_constants:
    :return:
    '''
    out_state = spacecraft_state()
    out_state.state_vec = safe_constants[0].dot(input_state.state_vec) * time + input_state.state_vec
    out_state.error_mode = 0

    return out_state

class RelativeOrbitEnv(gym.Env):
    """
    Define a simple orbit environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("RelOrbitEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.step_timestep = 60*60. #  Default "mode" timestep is 1 hour
        self.dt = 1.0 # Propagation timestep w/in a mode is 1 second
        self.curr_step = -1
        self.max_length = 100 # Specify a maximum number of timesteps

        # Define what the agent can do
        # Defines the number of discrete spacecraft modes.
        self.action_space = spaces.Discrete(3)

        self.cost_modifier = -0.1

        self.A = np.array([0.001])
        self.B = np.array([1])
        self.C = np.array([1])
        self.D = np.array([0])
        self.K = np.array([1.01])
        self.L = np.array([1.01])

        obs_stm = np.reshape(np.array([[self.A, np.zeros(self.A.shape)],
                           [np.zeros(self.A.shape), self.A - self.L.dot(self.C)]]), [2,2])
        ctrl_stm = np.reshape(np.array([[self.A - self.B*self.K, self.B*self.K], [np.zeros(self.A.shape), self.A]]), [2,2])
        error_stm = np.reshape(np.array([[self.A,np.zeros(self.A.shape)],[np.zeros(self.A.shape), self.A]]), [2,2])
        safe_stm = np.reshape(np.array([[0.0001*self.A,np.zeros(self.A.shape)],[np.zeros(self.A.shape), 0.0001*self.A]]), [2,2])



        self.obs_mode_constants = [obs_stm, 0.9]
        self.control_mode_constants = [ctrl_stm, 0.9]

        self.error_mode_constants = [error_stm, 1.0]
        self.safe_mode_constants = [safe_stm, 0.0]

        #self.obs_mode_constants = [1.0, -10.0, 0.9]
        #self.control_mode_constants = [-10.0, 1.0, 0.9]
        #self.error_mode_constants = [2.0, 2.0, 1.0]
        #self.safe_mode_constants = [0.1,0.1,0]

        self.curr_state = spacecraft_state()
        self.init_state = np.array([10000,10000])
        self.curr_state.state_vec = self.init_state

        # Observation is the current "true" state error.
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([100000.,  # pick a big number
                         ])
        self.observation_space = spaces.Box(low, high)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.episode_over = False

    def _step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.curr_step > self.max_length:
            self.episode_over = True
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.episode_over, {}

    def _take_action(self, action):

        self.action_episode_memory[self.curr_episode].append(action)

        if self.curr_state.error_mode == 1:
            #   check to see if we entered safe mode
            if action == 2:
                self.curr_state = safe_mode(self.curr_state, self.step_timestep, self.safe_mode_constants)
            else:
                self.curr_state = error_mode(self.curr_state, self.step_timestep, self.error_mode_constants)
        else:
            if action == 0:
                consts = self.obs_mode_constants
            elif action == 1:
                consts = self.control_mode_constants
            elif action == 2:
                consts = self.safe_mode_constants
            else:
                print "Action not found. Using safe mode constants:"
                consts = self.safe_mode_constants

            self.curr_state = operational_mode(self.curr_state, self.step_timestep, consts)

        remaining_steps = self.max_length - self.curr_step

    def _get_reward(self):
        """Reward is given for a sold banana."""
        return self.cost_modifier * (self.curr_state.state_vec[0])**2.0

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_state = spacecraft_state()
        self.curr_state.state_vec = self.init_state
        self.action_episode_memory.append([])

        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.curr_state
        return ob