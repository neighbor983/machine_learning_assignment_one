import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_List = [   { 'x': 2.0, 'y': 5.1},
                { 'x': 2.5, 'y': 6.1},
                { 'x': 3.0, 'y': 6.9},
                { 'x': 3.5, 'y': 7.8},
                { 'x': 4.0, 'y': 9.2},
                { 'x': 4.5, 'y': 9.9},
                { 'x': 5.0, 'y': 11.5},
                { 'x': 5.5, 'y': 12.0},
                { 'x': 6.0, 'y': 12.8}];

def simple_linear(theta0, theta1, x):
    '''
    description:
        takes in the thetas and the x_i and returns the predicted y for a simple linear regression
    params:
        theta0 = number
        theta1 = number    
        x = number
    output:
        number
    '''
    return theta0 + theta1 * x;

def cost_function(theta0, theta1, alpha, m, dataList):
    '''
    description:
        get the cost for a given model and thetas
    params:
        theta0 = number
        theta1 = number
        alpha = number
        m = number
        dataList = list of objects with 'x' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList: 
        x = row['x'];
        y = row['y'];
        summation += (simple_linear(theta0, theta1,x) - y)**2;
    cost = (.5*m) * summation;
    return cost;

def theta0_gradient(theta0, theta1, alpha, m, dataList) : 
    '''
    description:
        Use gradient descent to generate the new theta0
    params:
        theta0 = number
        theta1 = number
        alpha = number
        dataList = list of objects with 'x' and 'y'
    output:
        number
    '''
    summation = 0.0;
    for row in dataList:  
        x = row['x'];
        y = row['y'];
        summation += (simple_linear(theta0, theta1, x) - y);
    theta0 = theta0 - ( alpha / m) * summation;
    return theta0

def theta1_gradient(theta0, theta1, alpha, m, dataList) : 
    '''
    description:
        Use gradient descent to generate the new theta1
    params:
        theta0 = number
        theta1 = number
        alpha = number
        dataList = list of objects with 'x' and 'y'
    output:
        number
    '''
    summation = 0.0
    for row in dataList: 
        x = row['x'];
        y = row['y'];
        summation += ((simple_linear(theta0, theta1, x) - y) * x)
    theta1 = theta1 - ( alpha / m ) * summation
    return theta1

#Initial variables
m = len(data_List);
theta0 = 1.5;
theta1 = 2.0;
alpha = .1;
count = 0.0;
runs = [];

j_theta = cost_function(theta0, theta1, alpha, m, data_List);
#print("J: " + str(j_theta));
count += 1;
runs.append({'run' : count, "J": j_theta});

theta0_new = theta0_gradient(theta0, theta1, alpha, m, data_List);
theta1_new = theta1_gradient(theta0, theta1, alpha, m, data_List);
theta0 = theta0_new;
theta1 = theta1_new;
j_theta_new = cost_function(theta0_new, theta1_new, alpha, m, data_List);
#print("J_new: " + str(j_theta_new));
count += 1;
runs.append({'run' : count, "J": j_theta});

while( j_theta > ( j_theta_new * 1.000001 ) ):
    theta0_new = theta0_gradient(theta0, theta1, alpha, m, data_List);
    theta1_new = theta1_gradient(theta0, theta1, alpha, m, data_List);
    theta0 = theta0_new;
    theta1 = theta1_new;
    j_theta = j_theta_new;
    j_theta_new = cost_function(theta0_new, theta1_new, alpha, m, data_List);
    count += 1;
    runs.append({'run' : count, "J": j_theta});
    #print("J: " + str(j_theta_new));
    #print("Thetaknot: " + str(theta0_new));
    #print("ThetaOne: " + str(theta1_new));

print('count: ' + str(count));

cost = [];
iteriation = [];

for run in runs:
    cost.append(run['J']);
    iteriation.append(run['run']);

plt.plot(iteriation, cost);
plt.xlabel('Runs');
plt.ylabel('J(theta) Cost');
plt.title('Problem 1 - J(theta) vs Runs');

plt.savefig("Problem1_Cost_vs_Runs.svg");