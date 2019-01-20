#Prerequisites
import numpy as np 
import pandas as pd 
from sklearn import linear_model 
regr=linear_model.LinearRegression()

#loading data
#may have to change this based on the current location of csv file
url="/home/raj1998/ML-for-Beginners-Practise/Admission_Predict.csv"

data=pd.read_csv(url)

array=data.values
#convering into numpy arrays for easy calculation and handling
X=np.array(array[:,1:8])
Y=np.array(array[:,8])

#fitting the test set data into LinearRegression model
regr.fit(X[:-100],Y[:-100])

print("Applying Validation set\n")
print("Predicted Value          Actual Value\n")

Y_predicted=np.array(regr.predict(X[301:]))


for i in range(99):
    print(Y[i+301])
   #I've tried many methods to print them in a single line but haven't been able to do so
    # If anyone figures out, please let me know how to do it
    print(Y_predicted[i])
    
# importing data to a csv file with required parameters
pd.DataFrame(Y[301:],Y_predicted).to_csv("/home/raj1998/Downloads/new.csv",index_label=["Actual values","Predicted Values"])
