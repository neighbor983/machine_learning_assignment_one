'''
Model Y as a linear function of X1and X2.  Assume that the learning rate is 0.01, and initial values of the parameters are [1, 1, 1].  
1.	Illustrate gradient descent algorithm by updating the parameters 3 iterations.  
2.	Code and run the algorithm till convergence.

You must submit the following
1.	Source code
2.	Output – Initial and final values of parameters; regression line and data; plots of J
3.	Your observations and conclusions .

'''

#Given 
data = [
     { 'x1': 0, 'x2':  1, 'y': 0.6 },
     { 'x1': 1, 'x2':  0, 'y': 2.4 },
     { 'x1': 1, 'x2':  1, 'y': 1.6 },
     { 'x1': 2, 'x2':  1, 'y': 3.4 },
     { 'x1': 1, 'x2':  2, 'y': 0.5}
    ];

learning_rate = .01;
theta_0 = 1;
theta_1 = 1;
theta_2 = 1;