# Exercise 1. Linear Regression with one variable.

"""In this part of this exercise, you will implement linear regression with one
variable to predict profits for a food truck. Suppose you are the CEO of a
restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.
You would like to use this data to help you select which city to expand
to next.
The file ex1data1.txt contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss."""
#Necessary library imported
from numpy import *
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
DIR_PATH = 'F:\Educational Stuff\Hebi\Machine Learning and Data Science\Machine Learning\Machine Learning Algorithm-My Work\Machine Learning-Andrw.Ng Exercises\Exercise 1\mlclass-ex1/'
#Ex. 1.0 return a Identity Matrix(Warm up exercise)
def ex1_0():
     A = eye(5);
     return A;

def plot(X, y):
    plt.plot(X, y,'*', markersize=5)
    plt.xlabel('Population of a city')
    plt.ylabel('Profit of a food truck in city')
    
def load_data():
    data = genfromtxt( DIR_PATH + "ex1data1.txt",delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = len(y) #Number of samples
    y = y.reshape(m,1) #make vector of Y(output) samples
    return X, y;

def plot_data():
    X, y = load_data();
    plot(X, y)
    plt.show(block=True)

#Gradient Descent Loop

def gradient_descent(X, y, theta, alpha, iterations):
    "Gardient Descent in simple loop(step by step)"
    grad = copy(theta)
    m = len(y)
    n = shape(X)[1]

    for counter in range(0, iterations): #iterations to train
        cum_sum = [0 for x in range(0, n)] #n num. of cum_sum

        for j in range(0, n):
            for i in range(0, m):
                term = (hypothesis(X[i], grad) - y[i])
                cum_sum[j] += X[i,j] * (term)
                
        # assign new values for each gradient, this should be separate fro above loop
        # in order to achieve simulataneous update effect
        for j in range(0, n):
            grad[j] = grad[j] -cum_sum[j] * (alpha / m)

    return grad

#Hypotheses

def hypothesis(X, theta):
    return X.dot(theta)


#Cost Function J(theta)
def computecost(X, y, theta):
    m = len(y)
    cumulative_sum = 0
    for i in range(0, m):
        cumulative_sum +=(hypothesis(X[i],theta) - y[i])**2 #sqaured error function
    cumulative_sum = (1.0/(2*m)) * cumulative_sum
    return cumulative_sum[0]




#Implementation

def ex1_2():
    X, y = load_data();
    m = len(y)
    X = c_[ones((m, 1)),X] #add theta0 coloumn for intercept
    theta = zeros((2,1)) #as there are two thetas theta0 and theta1 which has to be organished in vector.
    iterations = 1500 #no. of interations to learn
    alpha = 0.01 #learning rate alpha
    cost = computecost(X, y, theta)
    theta = gradient_descent(X, y, theta, alpha, iterations) 
    print cost
    print theta
    print "Predictions: "
    predict = array([1,3.5]).dot(theta)
    predict1 = array([1,7]).dot(theta)
    print predict[0], predict1[0]

    plot(X[:, 1], y)
    plt.plot(X[:, 1],X.dot(theta),'b-')
    plt.show(block=True)
    plt.close()

#Meshgrid
def ex1_3():
     X, y = load_data();
     m = len(y)
     X = c_[ones((m, 1)),X]

     theta0_vals = linspace(-10,10,100)
     theta1_vals =linspace(-4,4,100)
     J_vals = zeros((len(theta0_vals),len(theta1_vals)), dtype=float64)
     for i, v0 in enumerate(theta0_vals):
          for j, v1 in enumerate(theta1_vals):
               theta = array((theta0_vals[i],theta1_vals[j])).reshape(2,1)
               J_vals[i,j] = computecost(X, y, theta)
     R, P = meshgrid(theta0_vals,theta1_vals)

     fig = plt.figure()
     ax = fig.gca(projection='3d')
     ax.plot_surface(R, P, J_vals)
     plt.show(block=True)

     fig = plt.figure()
     plt.contourf(R,P,J_vals,logspace(-2, 3, 20))
     plt.plot(theta[0],theta[1],'ro',markersize=10)
     plt.show(block=True)
     plt.close('all')



def main():
    print ex1_0();
    #plot_data()
    ex1_2()
    ex1_3()



if __name__ == '__main__':
    main()
