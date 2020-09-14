import re
import os
import status
import operator
import math
import numpy as np


import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create some test data
dx = 0.01
X  = np.arange(-2, 2, dx)
Y  = np.exp(-X ** 2)

# Normalize the data to a proper PDF
Y /= (dx * Y).sum()

# Compute the CDF
CY = np.cumsum(Y * dx)

# Plot both
plot(X, Y)
plot(X, CY, 'r--')

show() 