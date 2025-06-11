import numpy as np

def relative_position(body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position

def calculate_angle(body, new_magnitude):
        '''calculate the relative angle between the original body vector and their current relative position in radians'''
        dot = np.dot(body.position, body.initial_vector)
        return ((np.arccos(dot / (body.initial_vector_magnitude * new_magnitude))))

def update_limit_distances(body, current_distance):
        if body.max_distance < current_distance: body.max_distance = current_distance
        if body.min_distance > current_distance: body.min_distance = current_distance