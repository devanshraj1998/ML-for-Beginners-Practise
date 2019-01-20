Topic-"Graduate Admissions"
Subtopic="Predicting admission from important parameters"

Here we have just applied a simple techique to implement Linear Regression

First of all the required libraries were imported
The .csv file was loaded via pandas library
The loaded file was converted into an nd.array using numpy library

The sklearn library has predefined methods for  LinearRegression(linearmodel.LinearRegression())
The data was divided into 'Train'and 'Validation' set 

.fit() method was used to fit the data and .predict() method was used to predict vales for the validation set

Further the obtained arrays were converted into csv format using pandas.

SOURCE-"https://www.kaggle.com/mohansacharya/graduate-admissions"
