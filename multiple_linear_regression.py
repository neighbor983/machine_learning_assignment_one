from plot_helper import scatter_3d_plot, surface_3d_plot, theta_run_plot

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

alpha = .01;
theta0 = 1;
theta1 = 1;
theta2 = 1;
m = len(dataList);
count = 0; 
runs = [];

j_theta = Cost_Function(theta0, theta1, theta2, alpha, m, dataList);
count += 1;
runs.append({'run' : count, "J": j_theta, "theta_0": theta0, "theta_1": theta1, "theta_2": theta2});

#First Iteraion
new_theta0 = Theta0_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta1 = Theta1_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta2 = Theta2_Gradient(theta0, theta1, theta2, alpha, m, dataList); 

j_theta = Cost_Function(theta0, theta1, theta2, alpha, m, dataList);
count += 1;
runs.append({'run' : count, "J": j_theta, "theta_0": theta0, "theta_1": theta1, "theta_2": theta2});

#Second Iteraion
theta0 = new_theta0;
theta1 = new_theta1;
theta2 = new_theta2;
new_theta0 = Theta0_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta1 = Theta1_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta2 = Theta2_Gradient(theta0, theta1, theta2, alpha, m, dataList); 

j_theta = Cost_Function(theta0, theta1, theta2, alpha, m, dataList);
count += 1;
runs.append({'run' : count, "J": j_theta, "theta_0": theta0, "theta_1": theta1, "theta_2": theta2});

#Third Iteraion
theta0 = new_theta0;
theta1 = new_theta1;
theta2 = new_theta2;
new_theta0 = Theta0_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta1 = Theta1_Gradient(theta0, theta1, theta2, alpha, m, dataList);
new_theta2 = Theta2_Gradient(theta0, theta1, theta2, alpha, m, dataList); 

j_theta_new = Cost_Function(theta0, theta1, theta2, alpha, m, dataList);
count += 1;
runs.append({'run' : count, "J": j_theta, "theta_0": theta0, "theta_1": theta1, "theta_2": theta2});

while(j_theta  > (j_theta_new * 1.00001) ):
    j_theta = j_theta_new;
    theta0 = new_theta0;
    theta1 = new_theta1;
    theta2 = new_theta2;
    new_theta0 = Theta0_Gradient(theta0, theta1, theta2, alpha, m, dataList);
    new_theta1 = Theta1_Gradient(theta0, theta1, theta2, alpha, m, dataList);
    new_theta2 = Theta2_Gradient(theta0, theta1, theta2, alpha, m, dataList); 
    j_theta_new = Cost_Function(theta0, theta1, theta2, alpha, m, dataList);
    count += 1;
    runs.append({'run' : count, "J": j_theta, "theta_0": theta0, "theta_1": theta1, "theta_2": theta2});
    
print("count: " + str(count));
print("theta0: " + str(new_theta0));
print("theta1: " + str(new_theta1));
print("theta2: " + str(new_theta2));
print("J(0): " + str(j_theta_new));

cost = [];
iteriation = [];
theta0s = [];
theta1s = [];
theta2s = [];

for run in runs:
    cost.append(run['J']);
    iteriation.append(run['run']);
    theta0s.append(run['theta_0']);
    theta1s.append(run['theta_1']);
    theta2s.append(run['theta_2']);


x1 = [0, 1, 1, 2, 1];
x2 = [1, 0, 1, 1, 2];
y = [.6, 2.4, 1.6, 3.4, .5];

scatter_3d_plot(x1, x2, y, 'Problem 2 \nTraining Data', "Problem2_Training_Data.svg");
title = "Surface Plot\n" + str(round(theta0, 3)) + " + " + str(round(theta1, 3)) + " * x1 " + str(round(theta2, 3)) + " * x2";
surface_3d_plot(theta0, theta1, theta2, x1, x2, y, title, "Problem2_Surface_Plot.svg");
theta_run_plot(iteriation, theta0s, 'Problem 2 \nTheta0 vs Runs', 'Problem2_Theta0_vs_Runs.svg');
theta_run_plot(iteriation, theta1s, 'Problem 2 \nTheta1 vs Runs', 'Problem2_Theta1_vs_Runs.svg');
theta_run_plot(iteriation, theta2s, 'Problem 2 \nTheta2 vs Runs', 'Problem2_Theta2_vs_Runs.svg');