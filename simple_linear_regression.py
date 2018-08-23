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
    
def Cost_Function(theta_0, theta_1, dataList):
    total = 0.0;
    for data in dataList:
        total = total + (Linear_Model(theta_0, theta_1, data['x']) - data['y']) ** 2;
    return total / ( 2.0 * len(dataList) );

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

def Theta0_Partial_Derivative(theta_0, theta_1, dataList):
    '''
    description: 
        takes in the thetas and a dataList and produce the partial derivative 
        with respect to theta_0 when using a simple linear model
    params:
        theta_0 = number
        theta_1 = number
        dataList = List of objects with an "x" and "y" values
    output:
        number
    '''
    total = 0.0;
    for data in dataList:
        total += (Linear_Model(theta_0, theta_1, data['x']) - data['y']);
    return total / len(dataList);

def Theta1_Partial_Derivative(theta_0, theta_1, dataList):
    '''
    description: 
        takes in the thetas and a dataList and produce the partial derivative 
        with respect to theta_1 when using a simple linear model
    params:
        theta_0 = number
        theta_1 = number
        dataList = List of objects with an "x" and "y" values
    output:
        number
    '''
    total = 0.0;
    for data in dataList:
        total += ((Linear_Model(theta_0, theta_1, data['x']) - data['y']) * data['x']);
    return total / len(dataList);

def Theta0_Gradient_Descent(alpha, old_theta0, partial_derivative):
    '''
    description:
        Use gradient descent to generate the new theta_naught
    params:
        alpha = number 
        old_theta0 = number 
        partial_derivative = number
    output:
        output
    '''
    return old_theta0 - ( alpha * partial_derivative);

def Theta1_Gradient_Descent(alpha, old_theta1, partial_derivative):
    '''
    description:
        Use gradient descent to generate the new theta_1
    params:
        alpha = number 
        old_theta1 = number 
        partial_derivative = number
    output:
        output
    '''
    return old_theta1 - ( alpha * partial_derivative);

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
theta_one = 2;
learning_rate = .05;
previous_cost = 10000000;
new_cost = 0;
count = 0

while(abs(previous_cost - new_cost) > .0001):
#for theta in range(1, 10000):
    previous_cost = new_cost;
    theta_naught = Theta0_Gradient_Descent(learning_rate, theta_naught, 
                    Theta0_Partial_Derivative(theta_naught, theta_one, training_data));
    theta_one = Theta1_Gradient_Descent(learning_rate, theta_naught, 
                    Theta1_Partial_Derivative(theta_naught, theta_one, training_data));
    print("theta_naught:" );
    print(theta_naught);
    print("theta_one: ");
    print(theta_one);
    new_cost = Cost_Function(theta_naught, theta_one, training_data);
    print("Cost: ");
    print(new_cost);
    print("Count: ");
    count+=1;
    print(count);