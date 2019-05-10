#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:45:22 2019

@author: manoj1511
"""

# Set the working directory to the folder containing traces
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

%matplotlib qt
fig = plt.figure()
ax = fig.gca(projection='3d')

# Change the range based on the points you are tracing 
for aisle in range(0,5,1):
    for row in range(0,5,1):
        for col in range(0,5,1):        
            index = ((aisle) * 511 * 511) + ((row) * 511) + (col);
            filename = "trace_" + str(index) + ".bin"
            trace = np.fromfile(filename, np.float32)
            trace_size = int(trace.size/3)
            if(trace_size > 100000):            # plot only traces that didn't go out of bounds 
                trace = trace.reshape(trace_size, 3)
                ax.plot(trace[:,0], trace[:,1], trace[:,2])
ax.legend()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.invert_zaxis()
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()