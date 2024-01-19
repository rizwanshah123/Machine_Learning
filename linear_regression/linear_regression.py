#Imported all the necessary library in the code that i have used to trian my liner regression model

#from sklearn library i have imported datasets and linear_model
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#in the below line i have imported Pandas which is useful for Data Manupilation and i have used it to convert the dataset into DataFrame
import pandas as pd
#the numpy is another powerful library used for working with arrays or reshaping the data
import numpy as np

# mathplotlib is another power library used to work with visualization of data
from matplotlib import pyplot as plt

#in this line i have called _openml from where i will get the dataset
data=datasets._openml
#in this line i have passed the name of the dataset and it is  available in the openml webiste
data=data.fetch_openml(name='Housing-Prices-in-London',version=1)

#in this line i have converted the data into Data Frame
data=pd.DataFrame(data.data)

#in this line i have reshaped the the X value that is Area_in_sq_ft to 2D array and the need is that LinearRegression accept 2D or more
#and it is indepandent variable
x=np.array(data['Area_in_sq_ft']).reshape(-1,1)
#in this line i have taken the price from from data which is dependent variable
#the price will be dependent on Area_in_sq_ft
y=data['Price']
#i have created the object of linerRegression model
reg=linear_model.LinearRegression()

#the fit  is a method commonly used to train a machine learning model.
reg.fit(x,y)
#the predict is a methos used for predecting the value based on the variable it trained
prediction=reg.predict(x)

mae = mean_absolute_error(y, prediction)
mse = mean_squared_error(y, prediction)
r2 = r2_score(y, prediction)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

print("the value of co Efficient is ",reg.coef_)

print("the value of Interceept is ",reg.intercept_)

#below is the scatter plot of area_in_sq_ft with respect to Price on Y-axis
plt.xlabel('area in square feet')
plt.ylabel('Price')
plt.scatter(x,y,marker='+',color='red',label='Data Points')
#used plot method to draw the best fit line on the given data
plt.plot(x,prediction,color='blue',linewidth=2, label='Regression')
plt.show()
