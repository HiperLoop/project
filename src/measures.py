"""measures.py provides the program with the necessary functions that can be used to calculate angle or distance between to objects 
or regarding the original position of the body."""

import numpy as np
from loaders import load_simulation_from_file, update_bodies_from_row,write_body_to_file_calc

class Kepler:

        def __init__(self, num_of_bodies = 1):
                self.angles = np.zeros((num_of_bodies, 3))

        def calculate_angle(self, body, new_magnitude):
                '''calculate the relative angle between the original body vector and their current relative position in radians'''
                dot = np.dot(body.position, body.initial_vector)
                return ((np.arccos(dot / (body.initial_vector_magnitude * new_magnitude))))

        def update_limit_distances(self, body, current_distance):
                if body.max_distance < current_distance: body.max_distance = current_distance
                if body.min_distance > current_distance: body.min_distance = current_distance

        def calculate_period(self, n, body, distance, current_step, time_step):
                self.angles[n][0] = self.calculate_angle(body, distance)
                if self.angles[n][0] >= self.angles[n][1] and self.angles[n][1] < self.angles[n][2]:
                        body.amount_period +=1
                        body.period = (body.period+(current_step - 1)/body.amount_period* time_step/body.amount_period)/(1+1/body.amount_period)
                        body.semimajor_axis = (body.min_distance + body.max_distance) / 2
                        if body.amount_period == 1: 
                                body.period *= 2
                self.angles[n][2] = self.angles[n][1]
                self.angles[n][1] = self.angles[n][0]

        def calculate_Kepler(self, bodies, current_step, time_step):
                """ print("calc calles")
                print(bodies)
                print(current_step)
                print(time_step) """
                for n, body in enumerate(bodies):
                        distance = np.linalg.norm(body.position)
                        self.update_limit_distances(body, distance)
                        self.calculate_period(n, body, distance, current_step, time_step)

        def from_file(self, file_name, dimension):
                bodies, data, sim_params = load_simulation_from_file(file_name, dimension)
                self.angles = np.zeros((len(bodies), 3))
                dim, precision, step, iterations = sim_params
                dim = int(dim)
                precision = int(precision)
                step = float(step)
                iterations = int(iterations)
                for body in bodies:
                        body.initial_vector = body.position
                        body.initial_vector_magnitude = np.linalg.norm(body.position)
                        body.min_distance = body.initial_vector_magnitude

                for i, row in enumerate(data):
                        update_bodies_from_row(bodies, row, int(dim))
                        self.calculate_Kepler(bodies, i+1, step)

                        if i % ((precision * iterations)//100) == 0:
                                print(f'{i / ((precision * iterations)//100)}% done')
                write_body_to_file_calc(bodies)

def relative_position(body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position        