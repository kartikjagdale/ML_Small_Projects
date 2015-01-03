#Machine Learning Excercise 1(with multi_variable) 'Using Normal Equation'
'''For Scientific computing'''
from numpy import *
#import scipy

'''For plotting'''
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

DIR_PATH = 'F:\Educational Stuff\Hebi\Machine Learning and Data Science\Machine Learning\Machine Learning Algorithm-My Work\Machine Learning-Andrw.Ng Exercises\Exercise 1\mlclass-ex1/'


def load_data():
    data = genfromtxt(DIR_PATH+'ex1data2.txt',delimiter = ',')
    X, y = data[:,0:2], data[:,2:3]
    return X, y

"""def plot(X, y):
    plt.plot(X, y,'*', markersize=5)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')"""
    
def normalEquation(X, y):
    return linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#Predicting using Normal Equation(NO need of feature scaling)
def ex1_multi():
    X, y = load_data()
    m = len(y)
    X = c_[ones((m,1)),X] #add intecept to X
    
    theta = normalEquation(X, y)
    # 1650 sq feet 3 bedroom house and first column parameter = 1
    test = array([1.0,1650.0,3.0])
    print 'Prediction:\n',test.dot( theta )


def main():
    print 'Machine Learning Exercise 1 with multi variables (Using NormalEquation)'
    ex1_multi();


if __name__=='__main__':
    main()











