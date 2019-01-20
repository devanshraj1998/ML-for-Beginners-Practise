####Prerequisites
import pandas as pd 
import numpy as np 

####reading the data
dataset=pd.read_csv("/home/raj1998/Downloads/Google/googleplaystore.csv")

####Observing data
#dataset.head()
#dataset.info()

####Dropping Nan
dataset=dataset.dropna(axis=0,how='any',inplace= False)

####Convering Category to numbers

Category_set=dataset["Category"].unique()
Category_Dictionary={}

for i in range(len(Category_set)):
        Category_Dictionary[Category_set[i]]=i+1

dataset["Category"]=dataset["Category"].map(Category_Dictionary)
#dataset.info()

####Size of APP
def size_kb(size):
    if 'M' in size:
        x=size[:-1]
        x=float(x)*1024
        return x
    elif 'k' in size:
        x=size[:-1]
        return float(x)
    else:
        return None
####None is Nan

####Replacing the Size
dataset["Size"]=dataset["Size"].map(size_kb)
####Filling in Nan, by the previos value in the data
dataset["Size"].fillna(method='ffill',inplace=True)
#dataset.info()

####Rating of an APP
def rating_conv(rating):
    if rating!=None:
        return float(rating)
    else:
        return None

####Replacing the Rating
dataset["Rating"]=dataset["Rating"].map(rating_conv)
####Filling Nan
dataset["Rating"].fillna(method='ffill',inplace=True)
#dataset.info()

####Convering Reviews into float
dataset["Reviews"] = dataset["Reviews"].astype(float)

####Genres of an app
Genres_set=dataset["Genres"].unique()
Genres_Dictionary={}

for i in range(len(Genres_set)):
        Genres_Dictionary[Genres_set[i]]=i+1

dataset["Genres"]=dataset["Genres"].map(Genres_Dictionary)


####Cleaning Prices
def clean_P(price):
    if price=='0':
        return 0
    else:
        x=price[1:]
        return float(x)

dataset["Price"]=dataset["Price"].map(clean_P)
#dataset.info()

####Cleaning Types
def typeof(type):
    if type=='Free':
        return 0
    else:
        return 1

dataset["Type"]=dataset["Type"].map(typeof)


####Converting Content Rating
ContentRating_set=dataset['Content Rating'].unique()
ContentRating_Dict={}
for i in range(len(ContentRating_set)):
    ContentRating_Dict[ContentRating_set[i]]=i+1

dataset["Content Rating"]=dataset["Content Rating"].map(ContentRating_Dict).astype(int)
#dataset.info()

####Cleaning installs
dataset['Installs'] = [int(i[:-1].replace(',','')) for i in dataset['Installs']]



####Dropping Unnecessary Labels
#dataset.info()
dataset=dataset.drop(['Current Ver','Android Ver','Last Updated'],axis=1)

####Using numpy to convert into mathematical matrices
X=dataset.drop(labels='App',axis=1)
#X.shape()
Y=dataset.Rating

####Converting into np array
X=np.array(X)
Y=np.array(Y)

####Importing sklearn
from sklearn import linear_model
regr=linear_model.LinearRegression()

####Findig out the size
length=Y.shape
####As length is returened as a tuple we need to covert it into int
K=length[0]

####Applying LinearRegression
####Without Mean Normalization
regr.fit(X[:-2000],Y[:-2000])

Predict_NO_Normalize=regr.predict(X[(K-2000):])
#Predict_NO_Normalize.shape

####Applyinng Built-in Normalizer
NX=dataset.drop(labels='App',axis=1)
NX=np.array(NX)


from sklearn.preprocessing import Normalizer
transformer=Normalizer().fit(NX)
NX=transformer.transform(NX)

####Applying LinearRegression With Mean Normalization
regr.fit(NX[:-2000],Y[:-2000])
Predict_Norm=regr.predict(NX[(K-2000):])



####Outputting the Data in CSV Format With Comparison
####Between two differernt modes
#pd.DataFrame(Y[(K-2000)],Predict_NO_Normalize,Predict_Norm).to_csv("/home/raj1998/Downloads/Normalization.csv",index_label=["Actual values","Predicted Values","Predicted Vales(Normalization)"])

new=pd.DataFrame(Predict_NO_Normalize)
new1=pd.DataFrame(Predict_Norm)
new=new.join(new1,lsuffix='_new',rsuffix='_new1')
new2=pd.DataFrame(Y)
new=new.join(new2,lsuffix='_new',rsuffix='_new2')



####IMporting CSV
pd.DataFrame(new).to_csv("/home/raj1998/Downloads/Normalization.csv",index_label=["S.No","Predicted Values","Predicted Vales(Normalization)","Actual values"])




