'''
Model Y as a linear function of X1and X2.  Assume that the learning rate is 0.01, and initial values of the parameters are [1, 1, 1].  
1.	Illustrate gradient descent algorithm by updating the parameters 3 iterations.  
2.	Code and run the algorithm till convergence.

You must submit the following
1.	Source code
2.	Output – Initial and final values of parameters; regression line and data; plots of J
3.	Your observations and conclusions .

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


'''
     model = theta_0 + theta_1 * x_1 + theta_2 * x_2`
'''
