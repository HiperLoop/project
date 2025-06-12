"""Initialises the Body class"""
import numpy as np

class Body:
    """Class representing a celestial body with mass, radius, position, and velocity."""

    dimension = 3  # Number of dimensions (3D space)

    def __init__(self, name, colour, mass, radius, position, velocity):
        self.name = name
        self.display_colour = colour
        self.mass = mass
        self.radius = radius
        self.min_distance = 0
        self.max_distance = 0
        self.initial_vector = np.array(position)
        self.initial_vector_magnitude = 0
        self.period = 0
        self.amount_period = 0
        self.semimajor_axis = 0
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass}, radius={self.radius}, position={self.position}, velocity={self.velocity})"