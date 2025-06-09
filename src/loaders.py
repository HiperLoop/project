import numpy as np
import random
from body import Body
from datetime import datetime, date
from scipy.spatial.transform import Rotation as R
import os

OBJECTS_PATH = "./objects/"
SIMULATIONS_PATH = "./simulations/"

def write_body_to_file():
    """Write a single body's data to the file."""
    name = input("Enter body name: ")
    mass = input("Enter body mass: ")
    diameter = input("Enter body diameter: ")
    dimensions = input("Enter body dimensions (e.g., 2 for 2D, 3 for 3D): ")
    dim_labels = ["X", "Y", "Z"]
    pos = []
    vel = []
    for i in range(int(dimensions)):
        pos.append(input(f"Enter {dim_labels[i]} position in Gm: "))
        vel.append(input(f"Enter {dim_labels[i]} velocity in km/s: "))
    for i in range(int(dimensions), 3):
        pos.append("0")
        vel.append("0")

    file = open(OBJECTS_PATH+'custom_objects.csv', "a")
    file.write(name + "," + mass + "," + diameter + "," + ",".join(pos) + "," + ",".join(vel) + "\n")

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

def load_body_from_planets(filename, dimension=3, indexes=None):
    """Load body data from a CSV file with the following format: 
    Name,Mass (10^24 kg),Radius (km),Perihelion (10^6 km),Max Orbital Velocity (km/s),Orbit Inclination (deg),Orbit Eccentricity"""
    text_bodies = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=None, encoding=None)[1:]
    bodies = []
    for i, row in enumerate(text_bodies):
        if not(indexes and i not in indexes):
            name = row[0]
            mass = row[1]
            radius = float(row[2])
            position = np.array([float(row[3]),0,0])
            max_velocity = np.array([0,float(row[4]),0])
            #Rotate the position according to the inclination and setting it somewhere random around the sun
            inclination_rotation = R.from_euler('y', float(row[5]), degrees=True)
            random_rotation = R.from_euler('z', np.random.rand()*360, degrees=True)
            position=random_rotation.apply(inclination_rotation.apply(position))
            velocity=random_rotation.apply(inclination_rotation.apply(max_velocity))
            
            colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            body = Body(name, colour, float(mass), float(radius), position[:dimension], velocity[:dimension])
            bodies.append(body)
    return bodies

def load_body_from_custom_csv(filename, dimension=3, indexes=None):
    """Load body data from a CSV file."""
    text_bodies = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=None, encoding=None)[1:]
    bodies = []
    for i, row in enumerate(text_bodies):
        if not(indexes and i not in indexes):
            name = row[0]
            mass = row[1]
            radius = float(row[2])/2
            colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            position = np.array([float(col) for col in row[3:3+dimension]])
            velocity = np.array([float(col) for col in row[6:6+dimension]])
            body = Body(name, colour, float(mass), float(radius), position, velocity)
            bodies.append(body)
    return bodies

def load_boadies_by_name(names, dimension=3):
    files = os.listdir(OBJECTS_PATH)
    bodies = []
    for filename in files:
        file_body_indexes = []
        body_names = np.genfromtxt(OBJECTS_PATH+filename, delimiter=',', dtype=str, encoding=None)[1:, 0]
        for name in names:
            if name in body_names:
                file_body_indexes.append(np.where(body_names == name)[0])

        loaded_bodies = []
        if filename == 'planets.csv':
            loaded_bodies = load_body_from_planets(filename, indexes=file_body_indexes, dimension=dimension)
        elif filename == 'custom_objects.csv':
            loaded_bodies = np.array(load_body_from_custom_csv(filename, indexes=file_body_indexes, dimension=dimension))
        bodies.extend(loaded_bodies)
    return bodies

def write_simulation_to_file_step(file, bodies):
    """Write the current state of the simulation to a CSV file. DOES NOT CLOSE FILE!"""
    dimension = Body.dimension
    write_row = ""
    for i, body in enumerate(bodies):
        position = body.position if dimension == 3 else np.append(body.position, 0)
        velocity = body.velocity if dimension == 3 else np.append(body.velocity, 0)
        write_row += (",".join([str(coord) for coord in np.append(position, velocity)])) + ("," if i != len(bodies)-1 else "")
    file.write(write_row + "\n")

def write_simulation_to_file_step_y(file, y):
    """Write the current state of the simulation to a CSV file. DOES NOT CLOSE FILE!"""
    dimension = Body.dimension
    number_of_bodies = y.shape[1] // (dimension*2)
    write_row = ("\n".join([",".join([",".join([str(number) for number in np.concatenate([y[i, j*dimension:(j+1)*dimension] if dimension == 3 else np.append(y[i, j*dimension:(j+1)*dimension], 0), y[i, j*dimension+dimension:(j+1)*dimension+dimension] if dimension == 3 else np.append(y[i, j*dimension+dimension:(j+1)*dimension+dimension], 0)])]) for j in range(number_of_bodies)]) for i in range(y.shape[0])]))
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
    file.write(",".join([f"{body.name},{body.mass},{body.radius},{body.display_colour[1:]},0,0" for body in bodies]) + "\n")
    file.write("#" + "x,y,z,vx,vy,vz," * len(bodies) + "\n")
    file.close()
    file = open(file_name, "a")
    write_simulation_to_file_step(file, bodies)
    file.close()
    return file_name

def load_simulation_from_file(filename, dimension=2):
    """Load simulation data from a CSV file."""
    text_bodies = np.genfromtxt(SIMULATIONS_PATH+filename, delimiter=',', dtype=float, encoding=None)
    #print(text_bodies.shape)
    bodies = []
    characteristics = np.genfromtxt(SIMULATIONS_PATH+filename, delimiter=',', dtype=str, encoding=None)[0]
    initial_vectors = text_bodies[1]
    vector_data = text_bodies[2:]
    vec_size = 6
    for i in range(len(characteristics) // vec_size):
        name = characteristics[i*vec_size]
        mass = characteristics[i*vec_size + 1]
        radius = float(characteristics[i*vec_size + 2])
        colour = "#" + characteristics[i*vec_size + 3]
        position = np.array([float(initial_vectors[i*vec_size + j]) for j in range(dimension)])
        velocity = np.array([float(initial_vectors[i*vec_size + j + dimension]) for j in range(dimension)])
        body = Body(name, colour, float(mass), float(radius), position, velocity)
        bodies.append(body)
    return bodies, vector_data