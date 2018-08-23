'''
a.	Model Y as a linear function of X. 
b.	Use gradient descent learning algorithm to learn model parameters.  Use an appropriate 
    learning rate and convergence criterion. Plot J for the learning duration.
'''



def Linear_Model(theta_0, theta_1, x_i):
    '''
    description:
        takes in the thetas and the x_i and returns the predicted y for a simple linear regression
    params:
        theta_0 = number
        theta_1 = number    
        xi = number
    output:
        number
    '''
    return theta_0 + theta_1 * x_i;
    
def Theta0_Mean_Squared_Error(theta_0, theta_1, dataList):
    '''
    description: 
        takes in the thetas and a dataList and produces the mean square error 
        for theta_0 when using a simple linear model
    params:
        theta_0 = number
        theta_1 = number
        dataList = List of objects with an "x" and "y" values
    output:
        number
    '''
    error_sum = 0.0;
    for data in dataList:
        error_sum += ( Linear_Model(theta_0, theta_1, data['x']) - data['y'] ) ** 2;
    return error_sum / len(dataList);

def Simple_Linear_Theta0_Cost_Fuction(theta_0, theta_1, dataList):
    '''
    description:
        takes in thetas and dataList and produces the cost function for theta_0
    params:
        theta_0 = number
        theta_1 = number
        dataList = List of objects with an "x" and "y" values
    output:
        number
    '''
    m = len(dataList);
    return Theta0_Mean_Squared_Error(theta_0, theta_1, dataList) / 2.0;
    
#Given Values
training_data = [   { 'x': 2.0, 'y': 5.1},
                    { 'x': 2.5, 'y': 6.1},
                    { 'x': 3.0, 'y': 6.9},
                    { 'x': 3.5, 'y': 7.8},
                    { 'x': 4.0, 'y': 9.2},
                    { 'x': 4.5, 'y': 9.9},
                    { 'x': 5.0, 'y': 11.5},
                    { 'x': 5.5, 'y': 12.0},
                    { 'x': 6.0, 'y': 12.8}];

#Intial Guesses
theta_naught = 1;
theta_one = 1;
learning_rate = .5;


print(Theta0_Mean_Squared_Error(theta_naught, theta_one, training_data));
print(Simple_Linear_Theta0_Cost_Fuction(theta_naught, theta_one, training_data));