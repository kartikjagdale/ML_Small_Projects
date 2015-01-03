"""
Predicting which Television Show will have more viewers Flash or Arrow.

About data:(input_data.csv):
Taken from wikipedia

Tutorial from DataAspirant site:
Link: https://dataaspirant.wordpress.com/2014/12/20/linear-regression-implementation-in-python/
"""

#Required Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model

# Function to get data
def get_data(filename):
	data = pd.read_csv(filename)
	X_flash = []
	y_flash = []
	X_arrow = []
	y_arrow = []

	for x1, y1, x2, y2 in zip(data['flash_episode'], data['flash_us_viewers'], data['arrow_episode'], data['arrow_us_viewers']):
		X_flash.append([float(x1)])
		y_flash.append([float(y1)])
		X_arrow.append([float(x2)])
		y_arrow.append([float(y2)])
	return X_flash, y_flash, X_arrow, y_arrow

# Function for Linear Model
def linear_model_main(X1, y1, X2, y2):
	regr1 = linear_model.LinearRegression()
	regr1.fit(X1, y1)
	predict_flash = regr1.predict(9)
	print "Prediction for No. of Viewers for next week Flash episode ",predict_flash[0][0]
	regr2 = linear_model.LinearRegression() 
	regr2.fit(X2, y2)
	predict_arrow = regr2.predict(9)
	print "Prediction for Viewers for next Flash episode ",predict_arrow[0][0]

	if predict_arrow > predict_flash:
		print "The TV series Flash will have more viewers for next week."
	else:
		print "The TV series Arrow will have more viewers for next week."
# Function Main
def main():
	X1, y1, X2, y2 = get_data('input_data.csv') # here X1, y1 = X_flash and y_flash and X2, y2 = X_arrow, y_arrow
	linear_model_main(X1, y1, X2, y2)


if __name__ == '__main__':
	main()
	
"""
Output:
Prediction for No. of Viewers for next week Flash episode  3.95266666667
Prediction for Viewers for next Flash episode  3.19244444444
The TV series Arrow will have more viewers for next week.
[Finished in 1.9s]

"""
