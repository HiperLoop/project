import numpy as np
import random
from body import Body
from simulation import Simulation
from animation import Animation

def load_body_from_csv(filename, dimension=2):
        """Load body data from a CSV file."""
        text_bodies = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)[1:]
        bodies = []
        for row in text_bodies:
            name = row[0]
            mass = row[1]
            distance = row[8]
            radius = float(row[2])/2
            velocity = row[12]
            colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            body = Body(name, colour, float(mass), float(radius), [float(distance)] + [float(0)]*(dimension-1), [float(0), float(velocity)] + [float(0)]*(dimension-2))
            bodies.append(body)
        return bodies

def load_body_from_custom_csv(filename, dimension=2):
    """Load body data from a CSV file."""
    text_bodies = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)[1:]
    bodies = []
    for row in text_bodies:
        name = row[0]
        mass = row[1]
        radius = float(row[2])/2
        colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        position = np.array([float(col) for col in row[3:3+dimension]])
        velocity = np.array([float(col) for col in row[6:6+dimension]])
        print(position, velocity)
        body = Body(name, colour, float(mass), float(radius), position, velocity)
        bodies.append(body)
    return bodies

def figure_eight_configureation():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('./objects/custom_objects.csv')[-3:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=3, norm=False, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=3, sim_duration=1000, frame_rate=100)

def figure_eight_configureation_3D():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('./objects/custom_objects.csv', dimension=3)[-3:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=3, norm=False, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=3, sim_duration=1000, frame_rate=100)

def solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('./objects/planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('./objects/custom_objects.csv')[0]
    bodies = [sun] + planets
    print("Loaded bodies:", bodies)
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=40, sim_duration=1000, frame_rate=100)

def solar_system_3D():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('./objects/planets.csv', dimension=3)
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('./objects/custom_objects.csv', dimension=3)[0]
    bodies = [sun] + planets
    print("Loaded bodies:", bodies)
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=40, sim_duration=1000, frame_rate=100)

def main():
    """Main function to run the simulation."""
    # Uncomment the desired simulation configuration
    #figure_eight_configureation()
    #figure_eight_configureation_3D()
    #solar_system()
    solar_system_3D()

main()