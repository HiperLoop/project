from loaders import *
from simulation import Simulation
from animation import Animation

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