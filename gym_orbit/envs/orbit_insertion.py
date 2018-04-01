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
import orbitalMotion as om
import RigidBodyKinematics as rbk
import numpy as np
from gym import spaces


def orbitAfterThrustMars(input_state, DV):
    '''
    Mode in which the spacecraft is rendered non-functional and state error increases.
    :param input_state: state of the s/c when thrust is commanded
    :param DV: scalar thrust commanded (positive DV is against the s/c velocity)
    :return: oe : Orbital Elements of the resulting orbit (instantiation of ClassicElements class)

    '''
    thrust = np.zeros(len(input_state))
    thrust[3:6] = -DV*input_state[3:6]/np.linalg.norm(input_state[3:6])

    new_state = input_state + thrust

    return om.rv2elem_parab(om.MU_MARS, new_state[0:3], new_state[3:6])

class spacecraft_state():
    def __init__(self):
        # type: () -> object
        #self.state_err = 0
        #self.est_err= 0
        self.state_vec = np.array([0,0])
        self.error_mode = 0


class OrbitInsertion(gym.Env):
    """
    Define a simple orbit environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("OrbitEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.step_timestep = 1. #  Default "mode" timestep is 1 second
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