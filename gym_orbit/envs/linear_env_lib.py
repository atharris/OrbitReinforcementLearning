#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate a mars entry scenario using linear state and estimation error growth models.
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces

class spacecraft_state():
    def __init__(self):
        # type: () -> object
        self.state_err = 0
        self.est_err= 0
        self.error_mode = 0

def operational_mode(input_state, time, ops_constants):
    '''
    :param input_state:
    :param time:
    :param obs_constants: length-3 list in the following order:
                                    [ obs_estimation_constant, obs_control_constant, obs_error_constant
    :return:
    '''
    out_state = spacecraft_state()
    out_state.state_err = abs(input_state.state_err + ops_constants[0] * time)
    out_state.est_err = abs(input_state.est_err + ops_constants[1] * time)
    error_draw = np.random.uniform(0,1,[1,])
    if error_draw > ops_constants[2]:
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
    out_state.state_err = abs(input_state.state_err + error_constants[0] * time)
    out_state.est_err = abs(input_state.est_err + error_constants[1] * time)
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
    out_state.state_err = abs(input_state.state_err + safe_constants[0] * time)
    out_state.est_err = abs(input_state.est_err + safe_constants[1] * time)
    out_state.error_mode = 0

    return out_state

class LinearOrbitEnv(gym.Env):
    """
    Define a simple orbit environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("OrbitEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.step_timestep = 1.*60.*60.0 #  Default "mode" timestep is 1 hour
        self.curr_step = -1
        self.max_length = 100 # Specify a maximum number of timesteps

        # Define what the agent can do
        # Defines the number of discrete spacecraft modes.
        self.action_space = spaces.Discrete(3)
        self.state_error = 0
        self.control_error = 0

        self.cost_modifier = -1

        self.obs_mode_constants = [1.0, -10.0, 0.9]
        self.control_mode_constants = [-10.0, 1.0, 0.9]
        self.error_mode_constants = [2.0, 2.0, 1.0]
        self.safe_mode_constants = [0.1,0.1,0]

        self.curr_state = spacecraft_state()

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
            else:
                print "Action not found."
                consts = self.safe_mode_constants

            self.curr_state = operational_mode(self.curr_state, self.step_timestep, consts)

        remaining_steps = self.max_length - self.curr_step
        time_is_over = (remaining_steps <= 0)

    def _get_reward(self):
        """Reward is given for a sold banana."""
        return self.cost_modifier * self.curr_state.state_err

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_state = spacecraft_state()
        self.action_episode_memory.append([])

        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.curr_state
        return ob