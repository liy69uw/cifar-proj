#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to unpack pickle files.

@author: Zhaoqi Li
"""

import pickle
import numpy as np

# with open('image-train', 'rb') as f:
#     X_train = pickle.load(f)
# with open('label-train', 'rb') as f:
#     y_train = pickle.load(f)

with open('image-test', 'rb') as f:
    X_test = pickle.load(f)

# np.savetxt("image-train.txt", X_train)
# np.savetxt("label-train.txt", y_train)

np.savetxt("image-test.txt", X_test)
