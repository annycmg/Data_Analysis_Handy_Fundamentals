import pandas as pd 

# read in data from workign directory (folder in top right)
# can read in from anywhere if full path is the pd.read_csv()
data = pd.read_csv('craigslistVehicles.csv')

#rows and columns returns (rows, columns)
data.shape

#returns the first x number of rows when head(num)
data.head()

#returns the last x number of rows when tail(num)
data.tail()

#returns an object with all of the column headers 
data.columns

#basic information on all columns 
data.info()

#gives basic statistics on numeric columns
data.describe()

#shows what type the data was read in.,
data.dtypes

#shows which values are null
data.isnull()

#shows which columns have null values
data.isnull().any()

#shows for each column the percentage of null values 
data.isnull().sum() / data.shape[0]

# for categorical variables 
#shows unique values that appear in the column 
data.unique()

#shows the counts for those unique values 
data.unique().value_counts()

data.cylinders.head(10)

############################################################# Graphing #######################################################
#histogram of year 
# data.year.hist() == data.year.plot(kind='hist')
data.year.hist()
data.year.hist(bins=100)

#bar chart of types 
data.type.value_counts().plot(kind='bar')

