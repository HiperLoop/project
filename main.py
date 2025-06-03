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

def relative_force(body1, body2):
    """Calculate the gravitational force exerted by body2 on body1."""
    g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    G = 12000  # Modified gravitational constant
    r_vector = relative_position(body1, body2)
    distance = np.linalg.norm(r_vector)
    
    if distance == 0:
        return np.zeros(body1.dimension)  # No force if bodies are at the same position
    
    force_magnitude = G * body1.mass * body2.mass / distance**2
    force_vector = (force_magnitude / distance) * r_vector
    return force_vector

def forces(bodies):
    forces = np.zeros((len(bodies), bodies[0].dimension))
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                forces[i] += relative_force(body1, body2)
                forces[j] -= relative_force(body1, body2)
    print("Forces calculated:", forces)
    return forces

def accelerations(bodies):
    accelerations = forces(bodies)
    for i, body in enumerate(bodies):
        accelerations[i] /= body.mass
    print("Accelerations calculated:", accelerations)
    return accelerations