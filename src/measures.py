import numpy as np

def relative_position(body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position

def calculate_angle(original_vector, bodies):
        '''calculate the relative angle between the original body vector and their current relative position in radians'''
        new_vector = relative_position(bodies[0], bodies[1])
        dot = np.dot(original_vector, new_vector)
        original_magnitude = np.linalg.norm(original_vector)
        new_magnitude = np.linalg.norm(new_vector)
        print((np.arccos(dot / (original_magnitude * new_magnitude))))
        return ((np.arccos(dot / (original_magnitude * new_magnitude))))