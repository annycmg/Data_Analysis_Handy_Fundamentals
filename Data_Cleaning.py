import pandas as pd 

data = pd.read_csv('craigslistVehicles.csv')
data.columns

data.describe()

#remove duplicate rows in dataset 
data.drop_duplicates(inplace= True)

#looking for nulls / % of nulls 
data.isnull().any()
data.isnull().sum()/ data.shape[0]

#remove columns with certain threshold of nulls
#threshold is the number of columns or rows without nulls 
thresh = len(data)*.6
data.dropna(thresh = thresh, axis = 1)
data.dropna(thresh = 21, axis = 0)

#inputing fillna() to nulls
data.odometer.fillna(data.odometer.median())
data.odometer.fillna(data.odometer.mean())

#everything lower or uppercase
data.desc.head()
data.desc.head().apply(lambda x: x.lower())
data.desc.head().apply(lambda x: x.upper())

#use replace()
data.cylinders = data.cylinders.apply(lambda x: str(x).replace('cylinders','').strip())
data.cylinders.value_counts()

#change data type 
data.cylinders = pd.to_numeric(data.cylinders, errors = 'coerce')

#outlier detection and normalization remove rows with > 99% / z score 
numeric = data._get_numeric_data()
# with no null values 
from scipy import stats
import numpy as np 

data_outliers = data[(data.price < data.price.quantile(.995)) & (data.price > data.price.quantile(.005))]
data_outliers.boxplot('price')

#remove duplcates, subset, keep, etc.
data.drop_duplicates(subset=[], keep='')

#histogram
data_outliers.price.hist()

#type of normalization 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data.cylinders.values.reshape(-1,1))
scaler.transform(data.cylinders.values.reshape(-1,1))