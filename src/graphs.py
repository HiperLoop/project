import matplotlib.pyplot as plt 
from measures import *
import numpy as np
from loaders import *
from animation import *

def load_body_from_csv(filename, dimension=2):
        """Load body data from a CSV file."""
        bodies = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=None, encoding=None)
        bodies = []
        for row in bodies:
            name = row[0]
            T = row[1]
            SMA = row[2]
            bodies.append(body)
        return bodies

def scplt(filename):
    x = np.array([10**i for i in range (7)])
    fig = plt.figure()
    
    plot_size = parameters.plot_size
    plot_dimensions = parameters.plot_axis_limits
    ax = self.fig.add_subplot(projection='3d' if self.dimension == 3 else None)
    fig.set_figheight(self.plot_size)
    fig.set_figwidth(self.plot_size)