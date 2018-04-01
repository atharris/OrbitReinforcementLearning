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


def resultOrbit(input_state, desired_orbit):
    '''
    Mode in which the spacecraft is rendered non-functional and state error increases.
    :param input_state: state of the s/c when thrust is commanded
    :param DV: scalar thrust commanded (positive DV is against the s/c velocity)
    :return: oe : Orbital Elements of the resulting orbit (instantiation of ClassicElements class)

    '''
    assert isinstance(desired_orbit, om.ClassicElements)
    assert np.len(input_state)==6

    r_des, v_des = om.elem2rv_parab(desired_orbit)

    r1 = np.linalg.norm(input_state[0:3])
    r2 = np.linalg.norm(r_des)
    DV = np.sqrt(om.MU_MARS/r1)*(np.sqrt(2*r2/(r1+r2))-1.)

    thrust = np.zeros(len(input_state))
    thrust[3:6] = -DV*input_state[3:6]/np.linalg.norm(input_state[3:6])

    new_state = input_state + thrust

    return om.rv2elem_parab(om.MU_MARS, new_state[0:3], new_state[3:6])
