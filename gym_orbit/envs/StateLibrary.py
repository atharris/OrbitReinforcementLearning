#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate a mars entry scenario using linear state and estimation error growth models.
"""

import numpy as np


class linear_states():
    '''
    Simple state holder for two-dimensional model.
    '''
    def __init__(self):
        self.state_vec = np.array([0,0])
        self.burns = 0
        self.error_mode = 0

class rv_state():
    '''
    Full state vector for nonlinear model.
    '''
    def __init__(self):
        self.state_vec = np.zeros([6,])
        self.burns = 0

class observed_state():
    '''
    Holder for estimated state and covariance. Uses rv_state to hold position, velocity.
    '''
    def __init__(self):
        self.state_vec = np.zeros([6,])
        self.burns = 0
        self.covariance = np.zeros([6,6])
