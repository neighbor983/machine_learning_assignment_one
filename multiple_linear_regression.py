import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
Model Y as a linear function of X1and X2.  Assume that the learning rate is 0.01, and initial values of the parameters are [1, 1, 1].  
1.	Illustrate gradient descent algorithm by updating the parameters 3 iterations.  
2.	Code and run the algorithm till convergence.

You must submit the following
1.	Source code
2.	Output – Initial and final values of parameters; regression line and data; plots of J
3.	Your observations and conclusions.

'''
def Linear_Model(theta0, theta1, theta2, x1, x2):
    '''
    description:
        takes in the thetas and the xs and returns the predicted y for a 
        multiple linear regression
    params:
        theta0 = number
        theta1 = number  
        theta2 = number  
        x1 = number
        x2 = number  
    output:
        number
    '''
    return theta0 + theta1 * x1 + theta2 * x2;
    
def Cost_Function(theta0, theta1, theta2, alpha, m, dataList):
    '''
    description:
        get the cost for a given model and thetas
    params:
        theta0 = number
        theta1 = number
        theta2 = number
        alpha = number
        m = number
        dataList = list of objects with 'x1', 'x2' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList: 
        x1 = row['x1'];
        x2 = row['x2'];
        y = row['y'];
        summation += (Linear_Model(theta0, theta1, theta2, x1, x2) - y)**2;
    cost = ( .5 * m ) * summation;
    return cost;
     
def Theta0_Gradient(theta0, theta1, theta2, alpha, m, dataList) : 
    '''
    description:
        Use gradient descent to generate the new theta0
    params:
        theta0 = number
        theta1 = number
        theta2 = number
        alpha = number
        m = number
        dataList = list of objects with 'x1', 'x2' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList:  
        x1 = row['x1'];
        x2 = row['x2'];
        y = row['y'];
        summation += (Linear_Model(theta0, theta1, theta2, x1, x2) - y);
    theta0 = theta0 - ( alpha / m) * summation;
    return theta0
    
def Theta1_Gradient(theta0, theta1, theta2, alpha, m, dataList) : 
    '''
    description:
        Use gradient descent to generate the new theta1
    params:
        theta0 = number
        theta1 = number
        theta2 = number
        alpha = number
        m = number
        dataList = list of objects with 'x1', 'x2' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList:  
        x1 = row['x1'];
        x2 = row['x2'];
        y = row['y'];
        summation += ( (Linear_Model(theta0, theta1, theta2, x1, x2) - y ) * x1);
    theta1 = theta1 - ( alpha / m) * summation;
    return theta1   

def Theta2_Gradient(theta0, theta1, theta2, alpha, m, dataList) : 
    '''
    description:
        Use gradient descent to generate the new theta2
    params:
        theta0 = number
        theta1 = number
        theta2 = number
        alpha = number
        m = number
        dataList = list of objects with 'x1', 'x2' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList:  
        x1 = row['x1'];
        x2 = row['x2'];
        y = row['y'];
        summation += ( (Linear_Model(theta0, theta1, theta2, x1, x2) - y ) * x2);
    theta2 = theta2 - ( alpha / m) * summation;
    return theta2   
    
#Given 
dataList = [
     { 'x1': 0, 'x2':  1, 'y': 0.6 },
     { 'x1': 1, 'x2':  0, 'y': 2.4 },
     { 'x1': 1, 'x2':  1, 'y': 1.6 },
     { 'x1': 2, 'x2':  1, 'y': 3.4 },
     { 'x1': 1, 'x2':  2, 'y': 0.5}
    ];

learning_rate = .01;
theta0 = 1;
theta1 = 1;
theta2 = 1;

'''
First Iteraions
'''
new_theta0 = theta0;
new_theta1 = theta1;
new_theta2 = theta2; 





theta_0_new = Theta_Gradient_Descent(learning_rate, theta_0, 
               Theta0_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
theta_1_new = Theta_Gradient_Descent(learning_rate, theta_1, 
               Theta1_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
theta_2_new = Theta_Gradient_Descent(learning_rate, theta_2, 
               Theta2_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
print('First Update: ');
print('theta0: ' + str(theta_0_new));
print('theta1: ' + str(theta_1_new));
print('theta2: ' + str(theta_2_new));

theta_0_new1 = Theta_Gradient_Descent(learning_rate, theta_0_new, 
               Theta0_Partial_Derivative(theta_0_new, theta_1_new, theta_2_new, training_data));
theta_1_new1 = Theta_Gradient_Descent(learning_rate, theta_1_new, 
               Theta1_Partial_Derivative(theta_0_new, theta_1_new, theta_2_new, training_data));
theta_2_new1 = Theta_Gradient_Descent(learning_rate, theta_2_new, 
               Theta2_Partial_Derivative(theta_0_new, theta_1_new, theta_2_new, training_data));
print('Second Update: ');
print('theta0: ' + str(theta_0_new1));
print('theta1: ' + str(theta_1_new1));
print('theta2: ' + str(theta_2_new1));

theta_0_new2 = Theta_Gradient_Descent(learning_rate, theta_0_new1, 
               Theta0_Partial_Derivative(theta_0_new1, theta_1_new1, theta_2_new1, training_data));
theta_1_new2 = Theta_Gradient_Descent(learning_rate, theta_1_new1, 
               Theta1_Partial_Derivative(theta_0_new1, theta_1_new1, theta_2_new1, training_data));
theta_2_new2 = Theta_Gradient_Descent(learning_rate, theta_2_new1, 
               Theta2_Partial_Derivative(theta_0_new1, theta_1_new1, theta_2_new1, training_data));
print('Thrid Update: ');
print('theta0: ' + str(theta_0_new2));
print('theta1: ' + str(theta_1_new2));
print('theta2: ' + str(theta_2_new2));


previous_cost = 10;
new_cost = 5;

while( (previous_cost - new_cost) / previous_cost > .00001 ):
     previous_cost = new_cost;
     theta_0 = new_theta0;
     theta_1 = new_theta1;
     theta_2 = new_theta2;
     new_theta0 = Theta_Gradient_Descent(learning_rate, theta_0, 
               Theta0_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
     new_theta1 = Theta_Gradient_Descent(learning_rate, theta_1, 
               Theta1_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
     new_theta2 = Theta_Gradient_Descent(learning_rate, theta_2, 
               Theta2_Partial_Derivative(theta_0, theta_1, theta_2, training_data));
     new_cost = Cost_Function(new_theta0, new_theta1, new_theta2, training_data);
     print("Cost: " + str(new_cost));

print('Final Update: ');
print('theta0: ' + str(theta_0));
print('theta1: ' + str(theta_1));
print('theta2: ' + str(theta_2));
print('Model: ' + str(theta_0) + ' + ' + str(theta_1) + ' * x1 + ' + str(theta_2) + ' * x2');