import numpy as np
from body import Body

def relative_position(body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position

def calculate_angle(original_vector, bodies):
   new_vector = relative_position(bodies[0], bodies[1])
   dot = np.dot(original_vector, new_vector)
   original_magnitude = np.linalg.norm(original_vector)
   new_magnitude = np.linalg.norm(new_vector)
   print((np.arccos(dot / (original_magnitude * new_magnitude)))/np.pi * 180)
   return ((np.arccos(dot / (original_magnitude * new_magnitude)))/np.pi * 180)