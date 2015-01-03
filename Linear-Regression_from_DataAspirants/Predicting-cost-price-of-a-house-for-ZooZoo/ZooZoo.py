"""
This is a Fun Project about preditcing thehousing prices for zoozoo.

About data set: ('input_data.csv'):
Square feet is the  Area of house.
Price is the corresponding cost of  that house.

Here X values is Square_feet and y value us price

Tutorial on DataAspirant Site
Link: https://dataaspirant.wordpress.com/2014/12/20/linear-regression-implementation-in-python/
"""

#Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# Function to get data
def get_data(filename):
	data =pd.read_csv(filename)
	X = []; y = []
	for single_square_feet , single_price_value in zip(data['square_feet'], data['price']):
		X.append([float(single_square_feet)])
		y.append([float(single_price_value)])
	return X, y


# Function For Fitting our data to Linear Model
def linear_model_main(X, y, predict_value):
	# Create a Linear Regression Object
	regr = linear_model.LinearRegression()
	regr.fit(X, y)
	predict_outcome = regr.predict(predict_value)
	predictions = {}
	predictions['intercept'] = regr.intercept_ # Y intercept
	predictions['coefficient'] = regr.coef_
	predictions['predicted_value'] = predict_outcome
	return predictions # i.e return preicted value for a particluar square_feet, theta0, theta1

# Function to show the results of linear fit model
def plot_results(X, y):
	plt.xlabel('Square_feet')
	plt.ylabel('Price')
	plt.title('Predicting Housing Price for ZooZoo\n (Linear Model)')
	plt.grid(True)
	plt.xlim([0, 1000])
	plt.ylim([0, 25000])
	#Create linear regression object
	regr = linear_model.LinearRegression()
	regr.fit(X, y)
	plt.scatter(X, y, color='blue')
	plt.plot(X, regr.predict(X), color='red', linewidth=4)
	
	#plt.xticks(())
	#plt.yticks(())
	plt.show()




# Function Main 
def main():
	X, y = get_data('input_data.csv')
	#print X
	#print y
	predictvalue = 700 # this is our sample test ('Square_feet')
	result = linear_model_main(X, y, predictvalue)
	print 'Intercept Value: ', result['intercept']
	print 'Coefficient Value', result['coefficient']
	print 'Predicted Price for %d Sqaure Feet House is %f '%(predictvalue,result['predicted_value'])
	plot_results(X, y)




if __name__ == '__main__':
	main()
