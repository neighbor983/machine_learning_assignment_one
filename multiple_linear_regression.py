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
     
def Cost_Function(theta0, theta1, theta2, dataList):
     total = 0.0;
     for data in dataList:
          total += ( ( Linear_Model(theta0, theta1, theta2, data['x1'], data['x2'] ) - data['y'] ) ** 2 )
     return total / len(dataList);     
     
def Theta_Gradient_Descent(alpha, old_theta, partial_derivative):
    '''
    description:
        Use gradient descent to generate the new theta
    params:
        alpha = number 
        old_theta = number 
        partial_derivative = number
    output:
        output
    '''
    return old_theta - ( alpha * partial_derivative );
    
def Theta0_Partial_Derivative(theta0, theta1, theta2, dataList):
     '''
     description: 
        takes in the thetas and a dataList and produce the partial derivative 
        with respect to theta_0 when using a multiple linear model
     params:
        theta_0 = number
        theta_1 = number
        theta_2 = number
        dataList = List of objects with "x1", "x2", and "y" values
     output:
        number
     '''
     total = 0.0;
     for data in dataList:
          total += ( Linear_Model(theta0, theta1, theta2, data['x1'], data['x2']) - data['y'] );
     return total / len(dataList);
     
def Theta1_Partial_Derivative(theta0, theta1, theta2, dataList):
     '''
     description: 
        takes in the thetas and a dataList and produce the partial derivative 
        with respect to theta_1 when using a multiple linear model
     params:
        theta_0 = number
        theta_1 = number
        theta_2 = number
        dataList = List of objects with "x1", "x2", and "y" values
     output:
        number
     '''
     total = 0.0;
     for data in dataList:
          total += ( ( Linear_Model(theta0, theta1, theta2, data['x1'], data['x2']) - data['y'] ) * data['x1'] );
     return total / len(dataList);     

def Theta2_Partial_Derivative(theta0, theta1, theta2, dataList):
     '''
     description: 
        takes in the thetas and a dataList and produce the partial derivative 
        with respect to theta_2 when using a multiple linear model
     params:
        theta_0 = number
        theta_1 = number
        theta_2 = number
        dataList = List of objects with "x1", "x2", and "y" values
     output:
        number
     '''
     total = 0.0;
     for data in dataList:
          total += ( ( Linear_Model(theta0, theta1, theta2, data['x1'], data['x2']) - data['y'] ) * data['x2'] );
     return total / len(dataList); 

#Given 
training_data = [
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
new_theta0 = theta_0;
new_theta1 = theta_1;
new_theta2 = theta_2; 

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