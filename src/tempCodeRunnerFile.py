used_system=input('Please enter the name of the system you want to use: ')
            if((input('Do you want to animate your simulation simoultaneosly? (y/n): '))=='y'):
                anim_param=ask_animation_parameters()
                animate=True
            else:
                animate=False
            pre_made_systems[used_system](animate, anim_param)