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
#Import Necessary library
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model
DIR_PATH = 'F:\Educational Stuff\Hebi\Machine Learning and Data Science\Machine Learning\Machine Learning Algorithm-My Work\Machine Learning-Andrw.Ng Exercises\Exercise 1\mlclass-ex1/'
#-------


def plot(X, y):
    plt.plot(X, y,'rx', markersize=5, color = 'black')
    plt.xlabel('Population of a city')
    plt.ylabel('profit of a food truck in city')

def load_data():
    data = genfromtxt(DIR_PATH+"ex1data1.txt",delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = len(y) #number of samples
    X = X.reshape(m, 1); y = y.reshape(m, 1)
    return X, y

def implementaion():
    X, y = load_data();
    regr = linear_model.LinearRegression() #creat regression model object
    print X.shape
    print y.shape
    #Train tyhe model using training sets
    regr.fit(X, y)
    #the coefficients
    print('Coefficients: ',regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"%mean((regr.predict(X)- y)**2))
    #Explained varaince score: 1 is perfect prediction
    print('Variance Score: %.2f'%regr.score(X, y))
    #Prediction
    #predict1 = array([3.5]).reshape(-1,1)
    print regr.predict(7)
    #Plot Data
    plot(X, y)
    plt.plot(X, regr.predict(X), color = 'blue', linewidth=3)
    plt.show()
    plt.close()


def main():
    print "Starting linear regression"
    implementaion()

if __name__=='__main__':
    main()
