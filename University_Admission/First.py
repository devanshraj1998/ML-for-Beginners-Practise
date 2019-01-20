#Prerequisites
import numpy as np 
import pandas as pd 
from sklearn import linear_model 
regr=linear_model.LinearRegression()

#loading data
url="/home/raj1998/ML-for-Beginners-Practise/Admission_Predict.csv"

data=pd.read_csv(url)

array=data.values

X=np.array(array[:,1:8])
Y=np.array(array[:,8])

regr.fit(X[:-100],Y[:-100])

print("Applying Validation set\n")
print("Predicted Value          Actual Value\n")

Y_predicted=np.array(regr.predict(X[301:]))
##Y=Y.tolist()
##Y_predicted=Y_predicted.tolist()

for i in range(99):
    print(Y[i+301])
    ##print("     ")
    print(Y_predicted[i])
    ##print("\n")

pd.DataFrame(Y[301:],Y_predicted).to_csv("/home/raj1998/Downloads/new.csv",index_label=["Actual values","Predicted Values"])
