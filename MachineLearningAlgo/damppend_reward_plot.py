'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
theta_plot_lim = np.pi
theta_dot_plot_lim = 8

theta = np.arange(-theta_plot_lim, theta_plot_lim, 0.05)

theta_dot_space = 2*theta_dot_plot_lim / len(theta)
theta_dot = np.arange(-theta_dot_plot_lim, theta_dot_plot_lim, theta_dot_space)

theta, theta_dot = np.meshgrid(theta, theta_dot)
reward = -((abs(theta)-np.pi)*2)**2 + -0.25*(theta_dot)**2 + 50

# Plot the surface.
surf = ax.plot_surface(theta, theta_dot, reward, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax.set_title('Reward as a function of theta and theta_dot for damping the pendulum')
ax.set_xlabel('theta, rad')
ax.set_ylabel('theta_dot, rad/s')
ax.set_zlabel('reward')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=6)

plt.show()
