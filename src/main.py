from systems import *
def ask_animation_parameters():
    """Asks the user for the animation parameters that should be used for running the Animation"""
    animation_parameters=Animation_parameters(plot_axis_limits=float(input('Please enter the length of the plot limits in the normed value (normally AU): ')), plot_dimension=int(input('Please enter how many dimensions you want to animate (If you simulate 3 Dimensions but only animate 2, the z-axis will be removed) (2/3): ')), frame_rate=int(input('Please enter the framerate as an Integer: ')))
    return(animation_parameters)
def user_input():
    # print program functionality
    print("Welcome! This is a program that simulates and animates the gravity of n bodies.")
    sOrA=input('Do you want to simulate from the Scratch or animate from a file (s/a)') 
    if(sOrA=='a'):
        animate_from_file(input('Please enter the file name: '), ask_animation_parameters())
    elif(sOrA=='s'):
        sim_param=Simualtion_parameters(dimension=int(input('Please enter the number of dimensions you want to simulate (2/3): ')), step_precision=int(input('Please enter the step precision: ')), step_time=float(input('Please enter the step time: ')))
        preOrOwn=input('Do you want to simulate one of the premade systems or choose objects out of the files? (pre/own)')
        if (preOrOwn=='pre'):
            print("The currently available systems are: ")
            for system in pre_made_systems:
                print (system[0])
            pre_made_systems([input('Please enter the name of the system you want to use: ')])
        elif(preOrOwn=='own'):
            planets=(input('please write the planets as they are named in one of the .csv files that you want to use seperated by a comma: ')).split(',')
            if((input('Do you want to animate your simulation simoultaneosly? (y/n): '))=='y'):
                anim_param=ask_animation_parameters
                system_from_user_input(planets, simualtion_parameters=sim_param, animation_parameters=anim_param, animate=True)
            else:
                system_from_user_input(planets, simulation_parameters=sim_param)
    

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
    user_input()
    
#write_body_to_file()

main()