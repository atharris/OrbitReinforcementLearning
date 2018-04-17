#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate a mars entry scenario using linear state and estimation error growth models.
"""

# 3rd party modules
import gym
import numpy as np
import scipy as sci
from gym import spaces
import StateLibrary as sl
import ActionLibrary as al
import orbitalMotion as om

def set_moi_ic():
    est_state = sl.observed_state()
    true_state = sl.rv_state()
    ref_state = sl.rv_state()

    mode_options = al.mode_options()
    mode_options.dt = 1.0
    mode_options.mode_length = 10.*60.0
    mode_options.mu = om.MU_MARS
    mode_options.j2 = 0#om.J2_MARS
    mode_options.rp = om.REQ_MARS
    mode_options.burn_number = 1
    mode_options.error_stm = np.exp(mode_options.dt*(-0.01*np.identity(6)))

    desiredElements = om.ClassicElements()
    desiredElements.a = 1000
    desiredElements.e = 0.01
    desiredElements.Omega = 1.
    desiredElements.omega = 1.
    desiredElements.i = 1.
    desiredElements.f = 1.
    mode_options.goal_orbel = desiredElements
    mode_options.insertion_mode = True

    true_orbel = om.ClassicElements()
    true_orbel.a = 100000.0
    true_orbel.e = 0.8
    true_orbel.i = 0.0
    true_orbel.omega = 0.0
    true_orbel.Omega = 0.0
    true_orbel.f = -2.

    ref_orbel = om.ClassicElements()
    ref_orbel.a = 100000.0
    ref_orbel.e = 0.8
    ref_orbel.i = 0.0
    ref_orbel.omega = 0.0
    ref_orbel.Omega = 0.0
    ref_orbel.f = -2.

    est_orbel = om.ClassicElements()
    est_orbel.a = 100000.0
    est_orbel.e = 0.8
    est_orbel.i = 0.0
    est_orbel.omega = 0.0
    est_orbel.Omega = 0.0
    est_orbel.f = -2.

    ref_state.state_vec[0:3], ref_state.state_vec[3:] = om.elem2rv(om.MU_MARS, ref_orbel)
    true_state.state_vec[0:3], true_state.state_vec[3:] = om.elem2rv(om.MU_MARS, true_orbel)
    est_state.state_vec[0:3], est_state.state_vec[3:] = om.elem2rv(om.MU_MARS, est_orbel)
    return ref_state, est_state, true_state, mode_options


class mars_orbit_insertion(gym.Env):
    """
    Define a simple orbit environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("MOI - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length = 60 # Specify a maximum number of timesteps

        #   Set up options, constants for this environment
        self.ref_state, self.est_state, self.true_state, self.mode_options = set_moi_ic()
        self.control_use = 0.0

        #   Set up cost constants
        self.state_cost = -1.0 * np.identity(6)
        self.control_cost = -0.05


        # Observation is the estimated mean, and the associated covariance matrix
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([100000.,  # pick a big number
                         ])
        self.observation_space = spaces.Box(low, high)

        self.action_space = spaces.Discrete(3)
        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False

    def _seed(self):
        return

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
        if self.curr_step >= self.max_length:
            self.episode_over = True
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        info = {'ref': self.ref_state,
                'truth': self.true_state}
        return ob, reward, self.episode_over, info

    def _take_action(self, action):

        self.action_episode_memory[self.curr_episode].append(action)

        if action == 0:
            #   OD Step
            self.est_state, self.ref_state, self.true_state = al.observationMode(self. est_state, self. ref_state,
                                                                              self.true_state, self.mode_options)
            self.control_use = 0.0

        if action == 1:
            #   Control Step
            self.est_state, self.ref_state, self.true_state, self.control_use = al.controlMode(self.est_state, self.ref_state,
                                                                                 self.true_state, self.mode_options)

        if action ==2:
            #   DV Thrust Step
            self.est_state, self.ref_state, self.true_state, self.control_use = al.thrustMode(self.est_state, self.ref_state,
                                                                                 self.true_state, self.mode_options)
            print 'Thrust cost : ' , self.control_use
        remaining_steps = self.max_length - self.curr_step

    def _get_reward(self):
        """
        Cost is associated with an LQR cost model:
        C = (state error)^2 * state_cost + (control effort used)^2 *control_cost

        """
        err_state = self.true_state.state_vec - self.ref_state.state_vec
        return np.inner(err_state, self.state_cost.dot(err_state)) + self.control_use**2.0 * self.control_cost

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.ref_state, self.est_state, self.true_state, self.mode_options = set_moi_ic()
        self.control_use = 0.0
        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0
        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = {'state':self.est_state,
            'control':self.control_use}
        return ob
