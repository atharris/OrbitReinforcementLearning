#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate a mars entry scenario using linear state and estimation error growth models.
"""

import numpy as np


class Spacecraft_state():
    def __init__(self):
        # type: () -> object
        #self.state_err = 0
        #self.est_err= 0
        self.state_vec = np.array([0,0])
        self.error_mode = 0