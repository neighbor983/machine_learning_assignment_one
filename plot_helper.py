from mpl_toolkits import mplot3d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def scatter_plot(x, y, title, filename):
    plt.scatter(x,y);
    plt.xlabel('x');
    plt.ylabel('y');
    plt.title(title);
    plt.savefig(filename);
    plt.close();
    
def scatter_plot_regression(x,y,y1, title, filename):
    plt.scatter(x,y, label = 'data points');
    plt.plot(x,y1, label = 'regression line \n1.065 + 1.992 * x');
    plt.xlabel('x');
    plt.ylabel('y');
    plt.title(title);
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True);
    plt.savefig(filename);
    plt.close();
    
def line_plot(runs, cost, title, filename):
    plt.plot(runs, cost);
    plt.xlabel('Runs');
    plt.ylabel('J(theta) Cost');
    plt.title(title);
    plt.savefig(filename);
    plt.close();
    
def surface_plot(x1, x2, y, title, filename):
    fig = plt.figure();
    ax = plt.axes(projection='3d');
    ax.scatter(x1, x2, y);
    
    plt.savefig(filename);
    plt.close()