# Importing the libraries
import numpy as np
import pandas as pd
import pickle as pkl


# Importing the dataset
dataset = pd.read_excel('ml/BloodGlucose.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# print(x)
# print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Exporting the Linear Regression Model using pickle
pkl.dump(regressor,open('model.pkl','wb'))



# Predicting the Test set results
y_pred=regressor.predict(x_test)
# z=np.array([256,0.5,15])
# z=z.reshape(1,-1)
# z_pred=regressor.predict(z)
# print(z_pred)


from sklearn import metrics
# print("Mean Absolute Error: "+str(metrics.mean_absolute_error(y_test, y_pred)))
# print("Mean Squared Error: "+str(metrics.mean_squared_error(y_test, y_pred)))
# print("Mean Root Squared Error: "+str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

# print("Accuracy: "+str(regressor.score(x,y)))

Mean_Absolute_Error=metrics.mean_absolute_error(y_test, y_pred)
Mean_Squared_Error=metrics.mean_squared_error(y_test, y_pred)
Mean_Root_Squared_Error=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
accuracy=regressor.score(x,y)






