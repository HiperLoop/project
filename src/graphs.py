import matplotlib.pyplot as plt 
import numpy as np
from loaders import load_plot_data_from_csv

FIGURE_PATH = './plots/'

def scplt(filename):
    object_names, plot_data, label_data = load_plot_data_from_csv(filename=filename, norm=True, exclude_Sun=True)
    P_squared = plot_data[:, 0]
    SMA_cubed = plot_data[:, 1]
    P_squared_label = label_data[:, 0]
    SMA_cubed_label = label_data[:, 1]
    bot = int(np.floor(np.log10(min(SMA_cubed))))
    top = int(np.ceil(np.log10(max(SMA_cubed)))) + 1
    print(bot)
    print(top)
    x = np.array([10**(i) for i in range(bot, top)])
    y = x
    #y = np.array([10**(i) for i in range (-2, 4)])
    fig = plt.figure()

    planets = fig.add_subplot()
    planets.plot(x, y, 'g--', zorder=0)
    planets.scatter(P_squared, SMA_cubed, c='r', marker='d', zorder=1)
    for i, txt in enumerate(object_names):
        planets.annotate(txt, (P_squared[i], SMA_cubed[i]), xytext=(P_squared_label[i], SMA_cubed_label[i]))
    planets.set_xscale("log")
    planets.set_yscale("log")   
    
    plt.savefig(FIGURE_PATH+filename[:-4]+"_figure.png", format='png')

    plt.show()

scplt('rogue.csv')