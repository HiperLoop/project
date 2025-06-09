from systems import *

def ask_animation_parameters():
    """Asks the user for the animation parameters that should be used for running the Animation"""
    animation_parameters=Animation_parameters(plot_axis_limits=float(input('Please enter the length of the plot limits in the normed value (normally AU): ')), plot_dimension=int(input('Please enter how many dimensions you want to animate (If you simulate 3 Dimensions but only animate 2, the z-axis will be removed) (2/3): ')), frame_rate=int(input('Please enter the framerate as an Integer: ')))
    return(animation_parameters)
def user_input():
    # print program functionality
    print("Welcome! This is a program that simulates and animates the gravity of n bodies.")
    if(input('Do you want to simulate from the Scratch or animate from a file (s/a)')=='a'):
       animate_from_file(input('Please enter the file name: '), ask_animation_parameters())
    

def main():
    """Main function to run the simulation."""
    # Uncomment the desired simulation configuration
    #figure_eight_configureation()
    #figure_eight_configureation_3D()
    #solar_system()
    #from_file_test('2025-06-06_14-34-28.csv')
    #from_file_test('2025-06-06_15-26-35.csv')
    #solar_system_3D()
    #comet_solar_system()
    #system_from_file("2025-06-05_16-01-10.csv", dimension=2) # 2D load test
    #system_from_file("2025-06-05_16-28-24.csv", dimension=2)
    #system_from_file("2025-06-05_16-01-10.csv", dimension=3) # 3D load test
    #figure_eight_configureation(False)
    #user_input()
    system_from_user_input(['Sun', 'Rocket', 'Earth', 'Mercury'], animate=True, dimension=2, simulation_parameters=Simulation_parameters(dimension=2), animation_parameters=Animation_parameters(plot_dimension=2, plot_axis_limits=4))
    
#write_body_to_file()

main()