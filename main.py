import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Body:
    """Class representing a celestial body with mass, radius, position, and velocity."""

    dimension = 2  # Number of dimensions (3D space)

    def __init__(self, name, mass, radius, position, velocity):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass}, radius={self.radius}, position={self.position}, velocity={self.velocity})"
    
    def relative_position(body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position