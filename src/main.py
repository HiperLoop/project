"""main.py includes everything that is connected with the user. When the main file is called, the program starts as intended."""

from systems import *

def ask_animation_parameters():
    """Asks the user for the animation parameters that should be used for running the Animation"""
    animation_parameters=Animation_parameters(plot_axis_limits=float(input('Please enter the length of the plot limits in the normed value (normally AU): ')), 
                                              frame_rate=int(input('Please enter the framerate as an Integer: ')))
    return(animation_parameters)
def user_input():
    """Works as the main function to guide the user throught the features of the program"""
    print("Welcome! This is a program that simulates and animates the gravity of n bodies.")
    sOrA=input('Do you want to simulate from scratch or from a file (s/f)') 
    if(sOrA=='f'):
        if((input('Do you want to animate your simulation? (y/n): '))=='y'):
            anim_param=ask_animation_parameters()
            anim_param.plot_dimension=int(input("How many dimensions do you want to animate? (2/3)"))
            animate_from_file(input('Please enter the file name: '), anim_param)
        else:
            calculate_kepler_from_file((input('Please enter the file name: ')))
    elif(sOrA=='s'):
        sim_param=Simulation_parameters()
        if(input('Do you want to use the standard simulation parameters? (y/n): ')=='n'):
            sim_param=Simulation_parameters(dimension=int(input('Please enter the number of dimensions you want to simulate (2/3): ')), 
                                            step_precision=int(input('Please enter the step precision: ')), 
                                            step_time=float(input('Please enter the step time: ')))
        preOrOwn=input('Do you want to simulate one of the premade systems or choose objects out of the files? (pre/own)')
        if (preOrOwn=='pre'):
            print("The currently available systems are: ")
            for system in pre_made_systems:
                print (system)
            used_system=input('Please enter the name of the system you want to use: ')
            anim_param=Animation_parameters()
            if((input('Do you want to animate your simulation simoultaneosly? (y/n): '))=='y'):
                anim_param=ask_animation_parameters()
                animate=True
            else:
                animate=False
            pre_made_systems[used_system](animate, anim_param)
        elif(preOrOwn=='own'):
            planets=(input('please write the planets as they are named in one of the .csv files that you want to use seperated by a comma: ')).split(',')
            if((input('Do you want to animate your simulation simoultaneosly? (y/n): '))=='y'):
                anim_param=ask_animation_parameters()
                anim_param.plot_dimension=int(sim_param.dimension)
                system_from_user_input(planets, dimension=sim_param.dimension, simulation_parameters=sim_param, animation_parameters=anim_param, animate=True)
            else:
                system_from_user_input(planets, dimension=sim_param.dimension, simulation_parameters=sim_param)

user_input()