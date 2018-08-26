from mpl_toolkits import mplot3d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def scatter_plot(x, y, title, filename):
    plt.scatter(x,y);
    plt.xlabel('x');
    plt.ylabel('y');
    plt.title(title);
    plt.savefig(filename);
    plt.close();
    
def scatter_plot_regression(x,y,y1, title, filename):
    plt.scatter(x,y, label = 'data points');
    plt.plot(x,y1, label = 'regression line \n1.044 + 1.997 * X');
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
    
def scatter_3d_plot(x1, x2, y, title, filename):
    fig = plt.figure();
    ax = plt.axes(projection='3d');
    ax.scatter(x1, x2, y);
    
    plt.savefig(filename);
    plt.close();
    
def surface_3d_plot(theta0, theta1, theta2, title, filename):
    fig = plt.figure();
    ax = plt.axes(projection='3d');
    def f(x1, x2, theta0, theta1, theta2):
        return theta0 + theta1 * x1 + theta2 * x2; 

    x1 = np.linspace(-6, 6, 30);
    x2 = np.linspace(-6, 6, 30);

    X1, X2 = np.meshgrid(x1, x2);
    Z = f(X1, X2, theta0, theta1, theta2);
    
    
    ax.plot_surface(x1, x2, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    plt.savefig(filename);
    plt.close();