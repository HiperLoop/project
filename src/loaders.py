import numpy as np
import random
from body import Body
from datetime import datetime, date

OBJECTS_PATH = "./objects/"
SIMULATIONS_PATH = "./simulations/"

def load_body_from_csv(filename, dimension=2):
        """Load body data from a CSV file."""
        text_bodies = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=None, encoding=None)[1:]
        bodies = []
        for row in text_bodies:
            name = row[0]
            mass = row[1]
            distance = row[9]
            radius = float(row[2])/2
            velocity = row[12]
            colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            body = Body(name, colour, float(mass), float(radius), [float(distance)] + [float(0)]*(dimension-1), [float(0), float(velocity)] + [float(0)]*(dimension-2))
            bodies.append(body)
        return bodies

def load_body_from_custom_csv(filename, dimension=2):
    """Load body data from a CSV file."""
    text_bodies = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=None, encoding=None)[1:]
    bodies = []
    for row in text_bodies:
        name = row[0]
        mass = row[1]
        radius = float(row[2])/2
        colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        position = np.array([float(col) for col in row[3:3+dimension]])
        velocity = np.array([float(col) for col in row[6:6+dimension]])
        body = Body(name, colour, float(mass), float(radius), position, velocity)
        bodies.append(body)
    return bodies
     
def write_simulation_to_file_step(file, bodies):
    """Write the current state of the simulation to a CSV file. DOES NOT CLOSE FILE!"""
    dimension = Body.dimension
    write_row = ""
    for body in bodies:
        position = body.position if dimension == 3 else np.append(body.position, 0)
        velocity = body.velocity if dimension == 3 else np.append(body.velocity, 0)
        write_row += (",".join([str(coord) for coord in np.append(position, velocity)])) + ","
    file.write(write_row + "\n")
     
def write_simulation_to_file_init(bodies):
    """"Initialize the .cvs file and write the header."""
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    today = date.today()

    file_name = f'{SIMULATIONS_PATH}{today}_{current_time}.csv'
    file = open(file_name, "w")
    file.write("# ================================================================================================\n")
    file.write("#  This file contains the simulation data for a simulation of gravity between n bodies \n")
    file.write("#  The first line contains the defining charachterisitics of the bodies with 6 columns for each\n")
    file.write("#  Every subsequent line contains their position and velocities \n")
    file.write("# ================================================================================================\n")
    file.write("#\n")
    file.write("#" + "name,mass,radius,colour,_,_," * len(bodies) + "\n")
    file.write(",".join([f"{body.name},{body.mass},{body.radius},{body.display_colour},0,0" for body in bodies]) + "\n")
    file.write("#" + "x,y,z,vx,vy,vz," * len(bodies) + "\n")
    file.close()
    file = open(file_name, "a")
    write_simulation_to_file_step(file, bodies)
    file.close()
    return file_name