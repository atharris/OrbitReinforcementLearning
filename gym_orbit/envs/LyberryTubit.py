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

