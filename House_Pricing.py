# ML-Project-8-House-Pricing-1

#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import missingno as msno
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# print heading - for display purposes only
def print_heading(heading):
    print('-' * 50)
    print(heading.upper())
    print('-' * 50)
    
#** Load Dataset**

train=pd.read_csv('//content/drive/MyDrive/train1.csv')
test=pd.read_csv('/content/drive/MyDrive/test1.csv')
submission = pd.read_csv('/content/drive/MyDrive/sample_submission.csv')

train["group"] = "train"
test["group"] = "test"

df = pd.concat([train, test], axis=0)

train.drop("group", axis=1, inplace=True)
test.drop("group", axis=1, inplace=True)

df.head()

#** Some Data Exploration**

pd.set_option('display.max_columns', None) # to display all columns
train.head()

test.head()

submission.head()

#Shape of data

print_heading('Shape of train and test data')
print(f'In train dataset there are {train.shape[0]} rows and {train.shape[1]} columns')
print(f'In test dataset there are {test.shape[0]} rows and {test.shape[1]} columns')

#Statical information

train.info()

test.info()

#Describe

print_heading('Statical description  of train data')
train.describe()

#Certainly! Here are some key inferences from the summary statistics of the train dataset: Numerical Features:

#LotFrontage: The average lot frontage is around 70, with a minimum of 21 and a maximum of 313. MasVnrArea: Most houses have a masonry veneer area close to zero, with a maximum of 1600. GarageYrBlt: The average garage construction year is around 1978, with a minimum of 1900 and a maximum of 2010. Living Area Features:

#GrLivArea: The average above-ground living area is approximately 1515 square feet, ranging from 334 to 5642. TotalBsmtSF: Total basement area varies, with an average of 1057 square feet and a maximum of 6110. Bathroom Features:

#BsmtFullBath: Most houses have either no basement full baths or one. FullBath: The number of full bathrooms varies, with an average of 1.57. Garage Features:

#GarageCars: The average number of cars that can fit in garages is approximately 1.77, ranging from 0 to 4. GarageArea: Garage area varies, with an average of 472.98 square feet. Outdoor Features:

#WoodDeckSF: Many houses have a wood deck, with an average area of 94.24 square feet. OpenPorchSF: The average open porch area is 46.66 square feet. Pool and Miscellaneous:

#PoolArea: Most houses do not have a pool, with an average pool area of 2.76 square feet. MiscVal: The average miscellaneous value is around 43.49, with a maximum of 15500. Sale Price:

#SalePrice: The average sale price is approximately 180,921,withaminimumof 34,900 and a maximum of $755,000. note: These inferences provide a quick overview of the dataset, highlighting key statistics for various features.

print_heading('Statical description of test data')
test.describe()

#Here are some key inferences from the summary statistics of the Test dataset: Numerical Features:

#LotFrontage: The average lot frontage is around 68.58, with a minimum of 21 and a maximum of 200. MasVnrArea: The average masonry veneer area is approximately 100.71, with a maximum of 1290. GarageYrBlt: The average garage construction year is around 1977, with a minimum of 1895 and a maximum of 2207. Living Area Features:

#GrLivArea: The average above-ground living area is approximately 1486 square feet, ranging from 407 to 5095. TotalBsmtSF: Total basement area varies, with an average of 1046 square feet and a maximum of 5095. Bathroom Features:

#BsmtFullBath: Most houses have either no basement full baths or one. FullBath: The number of full bathrooms varies, with an average of 1.57. Garage Features:

#GarageCars: The average number of cars that can fit in garages is approximately 1.77, ranging from 0 to 5. GarageArea: Garage area varies, with an average of 472.77 square feet. Outdoor Features: Outdoor Features:

#WoodDeckSF: Many houses have a wood deck, with an average area of 93.17 square feet. OpenPorchSF: The average open porch area is 48.31 square feet. Pool and Miscellaneous:

#PoolArea: Most houses do not have a pool, with an average pool area of 1.74 square feet. MiscVal: The average miscellaneous value is around 58.17, with a maximum of 17000. Sale Price:

#This dataset doesn't include the "SalePrice" column, which was present in the first dataset.

#Shape of Datasets

# Checking the number of rows and columns
#Train
num_train_rows, num_train_columns = train.shape
#Test
num_test_rows, num_test_columns = test.shape

# Submission
num_submission_rows, num_submission_columns = submission.shape

#Printing the number of rows and columns.
print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Submission Data:")
print(f"Number of Rows: {num_submission_rows}")
print(f"Number of Columns: {num_submission_columns}")

# Null Values in Train
train_null = train.isnull().sum().sum()

#Null Count in Test
test_null = test.isnull().sum().sum()

#null Count in Submission
submission_null = submission.isnull().sum().sum()

print(f'Null Count in Train: {train_null}')
print(f'Null Count in Test: {test_null}')
print(f'Null Count in Submission: {submission_null}')

"Our training and testing datasets contain null values, which present a significant challenge."

#Duplicates Values of Datasets

# Count duplicate rows in train_data
train_duplicates = train.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test.duplicated().sum()

# Count duplicate rows in original_data
submission_duplicates = submission.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in test_data: {submission_duplicates}")

#Exploratary Data Analysis

#Deal with Missing values

missing_test  = test.copy()
missing_train  = train.copy()

missing_train.info()

#Training Data (df_train) Shape

# Training Data Shape
missing_train.shape

# Lets check the columns having categorical and object Datatype
# Initialize a counter for numeric columns
cat_columns = 0

# Iterate over column names and check data types
for column in missing_train.columns:
    if missing_train[column].dtype in ['category', 'object']:
        cat_columns += 1

# Print the total number of numeric columns
print("Total catorical columns:", cat_columns)

# Categorical columns in the Training Data
cat_columns=missing_train.select_dtypes(include=['category','object']).columns
cat_columns

#Lets check the columns having int and Float datatype

# Lets check the Number of columns Having int or Float Datatype
# Initialize a counter for numeric columns
numeric_columns = 0

# Iterate over column names and check data types
for column in missing_train.columns:
    if missing_train[column].dtype in ['int64', 'float64']:
        numeric_columns += 1

# Print the total number of numeric columns
print("Total numeric columns:", numeric_columns)

# Numerical columns( having int or Float datatype)
missing_train.select_dtypes(include=['float','int']).columns

train.columns

# Display Maximum Rows and columns of Training Data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Concatition data

missing_train['train']  = 1
missing_test['train']  = 0
df = pd.concat([missing_train, missing_test], axis=0,sort=False)

print_heading('Missing values in train and test data')
df.isnull().sum().sort_values(ascending=False).head(35)

print('shape of data',df.shape)

def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple.
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(df)

#Missing Data Visulization

# Assuming your DataFrame is named 'df'
plt.figure(figsize=(10, 5))
plt.title('Missing Data Visualization', fontsize=16, fontweight='bold')

# Create a missing data bar chart
msno.bar(df, color='blue')

# Show the plot
plt.show()

#Some data precessing for imputation of missing values

#Percentage of NAN Values
NAN = [(c, df[c].isna().mean()*100) for c in df]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN = NAN[NAN.percentage > 50]
NAN.sort_values("percentage", ascending=False)

#Dropping the columns

df=df.drop(['PoolQC','MiscFeature','Alley','Fence','MasVnrType'],axis=1)
#I am dropping these columns because they have missing values exceeding 50%.

#NUMERICAL DATA and CATEGORICAL DATA

print_heading('Numerical and cetegorical features in  data')
object_df = df.select_dtypes(include=['object'])
numerical_df =df.select_dtypes(exclude=['object'])

#datatype
print(object_df.dtypes)

print_heading('Numerical features in data')
print(numerical_df.dtypes)

#Treatment categorical data

print_heading('Categorical features of Missing values in data')
#Number of null values in each feature
null_counts = object_df.isnull().sum().sort_values(ascending=False)
null_counts.head(20)

#Fill the following columns with "None" (refer to the data description for details):

#BsmtQual

#BsmtCond

#BsmtExposure

#BsmtFinType1

#BsmtFinType2

#GarageType

#GarageFinish

#GarageQual

#FireplaceQu

#GarageCond

#Fill the remaining features with their respective most frequent values.

columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
object_df[columns_None]= object_df[columns_None].fillna('None')

print_heading('After Treating Missing values in data')
object_df.isnull().sum().sort_values(ascending=False).head(10)

#I am addressing the missing values in categorical features through imputation.

#Treatment Numerical Feature

print_heading('Numerical features in data')
#Number of null values in each feature
NUll_val = numerical_df.isnull().sum().sort_values(ascending=False)
NUll_val.head(20)

#Fill GarageYrBlt and LotFrontage with appropriate values.

#Fill the rest of the columns with 0.

print((numerical_df['YrSold']-numerical_df['YearBuilt']).median())
print(numerical_df["LotFrontage"].median())

#So we will fill the year with 1979 and the Lot frontage with 68

numerical_df['GarageYrBlt'] = numerical_df['GarageYrBlt'].fillna(numerical_df['YrSold']-35)
numerical_df['LotFrontage'] = numerical_df['LotFrontage'].fillna(68)

#Fill rest with zero 0

numerical_df= numerical_df.fillna(0)

numerical_df.isnull().sum().sort_values(ascending=False).head(10)

#Milestone: Finally impute Numerical values

#What's left of the Missing Values

#Navigating Special Values

#Upon closer examination of the evidence datasets, I found the unspecified clues were cataloged as "NaN" rather than the more descriptive "NA" label. While models can analyze numeric unknowns, interpreting context from categorical unknowns becomes ambiguous.

#I've reclassified all "NaN" clues as the clearer "NA" tag for non-existent attributes. This ensures any analytical tools I employ will treat undisclosed clues as meaningful categorical variables that represent potential attribute categories, rather than simply missing values.

#First, let us gather all these features that we have discoverd in one place

features_with_na = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
               
#Transforming "NA"s into Feature Gold

#Upon closer examination of the evidence datasets, I found the unspecified clues were cataloged as "NaN" rather than the more descriptive "NA" label. While models can analyze numeric unknowns, interpreting context from categorical unknowns becomes ambiguous.

#I've reclassified all "NaN" clues as the clearer "NA" tag for non-existent attributes. This ensures any analytical tools I employ will treat undisclosed clues as meaningful categorical variables that represent potential attribute categories, rather than simply missing values.

#First, let us gather all these features that we have discoverd in one place

#Now replace the NaN values with NA as I have explained above.

#To clarify:

#I am proactively classifying NA as meaningful attribute categories. This allows models to analyze NA as potential value options for features. Rather than opaque NaN, NA represents categorical information.

# replace all values of NaN with None
for feature in features_with_na:
    train[feature].fillna('NA', inplace=True)
    test[feature].fillna('NA', inplace=True)
#Now I will check if there any numerical missing values

numerical_columns = [col for col in train.columns if train[col].dtype != 'object']
# Get all catergorical columns
categorical_columns = [col for col in train.columns if train[col].dtype == 'object']

def count_missing_values(df, set='Train'):
    """
    This function checks missing values in the data
    """
    missing_value_counts_df = df.isnull().sum()
    missing_value_counts_df = missing_value_counts_df[missing_value_counts_df > 0].sort_values(ascending=False)

    # calculate percentage of missing values
    missing_value_percentage_df = round(missing_value_counts_df * 100 / len(df), 2).astype(str) + ' %'

    # concat missing count and percentage
    missing_values = pd.concat([missing_value_counts_df, missing_value_percentage_df], axis=1, keys=['Missing Values', 'Percent'])

    #missing_values = pd.DataFrame({set: missing_value_counts_df})
    return missing_values

# Display  all rows in the Test data that contain at least one missing value.
print_heading("Missing Values in Train Data")
missing_value_counts = count_missing_values(train[numerical_columns])
missing_value_counts

#Test Data Missing Values

# Display  all rows in the Test data that contain at least one missing value.
print_heading("Missing Values in Test Data")
missing_value_counts = count_missing_values(test, set='Test')
missing_value_counts

#Low Variance Columns

import matplotlib.pyplot as plt
import seaborn as sns

# Set up subplots for bar plots
plt.figure(figsize=[20, 10])

# Plot and display value counts for 'Utilities'
plt.subplot(2, 4, 1)
object_df['Utilities'].value_counts().plot(kind='bar')
plt.title('Utilities')
plt.xlabel('Values')
plt.ylabel('Count')

plt.subplot(2, 4, 2)
object_df['Street'].value_counts().plot(kind='bar')
plt.title('Street')
plt.xlabel('Values')
plt.ylabel('Count')

plt.subplot(2, 4, 3)
object_df['Condition2'].value_counts().plot(kind='bar')
plt.title('Condition2')
plt.xlabel('Values')
plt.ylabel('Count')
plt.subplot(2, 4, 4)
object_df['RoofMatl'].value_counts().plot(kind='bar')
plt.title('RoofMatl')
plt.xlabel('Values')
plt.ylabel('Count')

plt.subplot(2, 4, 5)
object_df['Heating'].value_counts().plot(kind='bar')
plt.title('Heating')
plt.xlabel('Values')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

#Due to the low variance columns we will drop them

object_columns_df = object_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)

#Why we drop Low variance Columns?

#Dropping low-variance columns helps eliminate features with minimal variability, streamlining the dataset and potentially improving model generalization by reducing noise and computational complexity.

# Lets Check the Missing values percentage present in the Training Data
missing_train.isnull().sum().sort_values(ascending=False)/len(missing_train)*100

train_num = train.select_dtypes(include = ['float64', 'int64'])
print(train_num.shape)
train_num.head()

#Plot the distribution for all the numerical features.

train_num.hist(figsize=(16, 20), bins=40, xlabelsize=8, ylabelsize=8);

#Multicollinearity

# check the correlation for columns => GarageArea & GarageCars with the target
print(train['GarageArea'].corr(train['GarageCars']))

print(train['GarageArea'].corr(train['SalePrice']))

print(train['GarageCars'].corr(train['SalePrice']))

# check the correlation for columns => YearBuilt & GarageYrBlt with the target
print(train['YearBuilt'].corr(train['GarageYrBlt']))

print(train['YearBuilt'].corr(train['SalePrice']))

print(train['GarageYrBlt'].corr(train['SalePrice']))

# check the correlation for columns => GrLivArea & TotRmsAbvGrd with the target
print(train['GrLivArea'].corr(train['TotRmsAbvGrd']))

print(train['GrLivArea'].corr(train['SalePrice']))

print(train['TotRmsAbvGrd'].corr(train['SalePrice']))

# check the correlation for columns => TotalBsmtSF & 1stFlrSF with the target
print(train['TotalBsmtSF'].corr(train['1stFlrSF']))

print(train['TotalBsmtSF'].corr(train['SalePrice']))

print(train['1stFlrSF'].corr(train['SalePrice']))

# check the correlation for columns => 2ndFlrSF & GrLivArea with the target
print(train['2ndFlrSF'].corr(train['GrLivArea']))

print(train['2ndFlrSF'].corr(train['SalePrice']))

print(train['GrLivArea'].corr(train['SalePrice']))

#Now we will search for columns that has a correlation more than 70% and drop one of them with the condition that the correlation with the target column (SalePrice) is smaller than another column

#missing_train

#Check for high and low cardinality

missing_train.select_dtypes('object').nunique().sort_values()

missing_train.info()

train["SalePrice"].quantile([0,0.25,0.50,0.75])

train["SalePrice_Range"] = pd.cut(train["SalePrice"],
                                 bins=np.array([-np.inf, 100, 150, 200, np.inf])*1000,
                                 labels=["0-100k","100k-150k","150k-200k","200k+"])

def find_col_dtypes(data, ord_th):

    num_cols = data.select_dtypes("number").columns.to_list()
    cat_cols = data.select_dtypes("object").columns.to_list()

    ordinals = [col for col in num_cols if data[col].nunique() < ord_th]

    num_cols = [col for col in num_cols if col not in ordinals]

    return num_cols, ordinals, cat_cols

num_cols, ordinals, cat_cols = find_col_dtypes(test, 20)

print(f"Num Cols: {num_cols}", end="\n\n")
print(f"Cat Cols: {cat_cols}", end="\n\n")
print(f"Ordinal Cols: {ordinals}")

plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.boxplot(x="SalePrice_Range", y=column, data=train,palette="twilight_shifted")
    plt.title(f"{column} Distribution")
    plt.tight_layout()

num_cols = train[num_cols].columns[train[num_cols].nunique() > 25]

plt.figure(figsize=(14,len(num_cols)*3))
for idx,column in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,idx+1)
    sns.histplot(x=column, hue="SalePrice_Range", data=train,bins=30,kde=True, palette="twilight_shifted")
    plt.title(f"{column} Distribution")
    plt.tight_layout()

train["SalePrice_Range"] = pd.cut(train["SalePrice"],
                                 bins=np.array([-np.inf, 100, 150, 200, np.inf])*1000,
                                 labels=["0-100k","100k-150k","150k-200k","200k+"])
                                 
Correlation

num_cols = train[num_cols].columns[train[num_cols].nunique() > 25]

plt.figure(figsize=(12,10))
corr=train[num_cols].corr(numeric_only=True)
mask= np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, fmt=".1f", linewidths=1, mask=mask, cmap=sns.color_palette("twilight_shifted"));

df_numeric = missing_train.select_dtypes(include='number').copy()
fig, axs = plt.subplots(10, 4,figsize = (35,35))

# Flatten the list of lists of subplots into a single list
axs = axs.ravel()

# Plot the boxplots for each column on the corresponding subplot
for i, column in enumerate(df_numeric):
  sns.boxplot(data=df_numeric[column], ax=axs[i])
  axs[i].set_title(column)
  
# Adjust the spacing between the subplots

skew_features = missing_train[df_numeric.columns].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

skew_index

missing_train

df_numeric = missing_train.select_dtypes(include='number').copy()

fig, axs = plt.subplots(10, 4,figsize = (35,35))

# Flatten the list of lists of subplots into a single list
axs = axs.ravel()

# Plot the boxplots for each column on the corresponding subplot
for i, column in enumerate(df_numeric):
  sns.boxplot(data=df_numeric[column], ax=axs[i])
  axs[i].set_title(column)
  
# Adjust the spacing between the subplots

missing_train.select_dtypes(include='object').columns

from sklearn.preprocessing import LabelEncoder
cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
missing_train[cols] = missing_train[cols].apply(LabelEncoder().fit_transform)

# Find the most frequent value in the 'Electrical' column
most_frequent_electrical = df['Electrical'].mode()[0]

# Impute missing values in 'Electrical' with the most frequent value
df['Electrical'].fillna(most_frequent_electrical, inplace=True)

# Check if 'Electrical' still has missing values
df['Electrical'].isnull().sum()

# Define the mappings for imputing categorical columns based on the data description
categorical_impute_map = {
    'PoolQC': 'N',
    'MiscFeature': 'N',
    'Alley': 'N',
    'Fence': 'N',
    'FireplaceQu': 'N',
    'GarageType': 'N',
    'GarageFinish': 'N',
    'GarageQual': 'N',
    'GarageCond': 'N',
    'BsmtExposure': 'N',
    'BsmtFinType2': 'N',
    'BsmtFinType1': 'N',
    'BsmtCond': 'N',
    'BsmtQual': 'N',
    'MasVnrType': 'N'
}

# Impute the categorical columns
for col, value in categorical_impute_map.items():
    train[col].fillna(value, inplace=True)

# Define the mappings for imputing numerical columns based on the data description
numerical_impute_map = {
    'MasVnrArea': 0
}

# Impute the numerical columns
for col, value in numerical_impute_map.items():
    train[col].fillna(value, inplace=True)

# Check for any remaining missing values
missing_data = missing_train.isna().sum()
missing_data = missing_data [missing_data > 0]

missing_train['LotFrontage'] = missing_train['LotFrontage'].fillna(0)
missing_train['Alley'] = missing_train['Alley'].fillna("No Alley Acess")
missing_train['MasVnrArea'] = missing_train['MasVnrArea'].fillna(df['MasVnrArea'].mean())
missing_train['LotFrontage'] = missing_train['LotFrontage'].fillna(0)
missing_train['BsmtQual'] = missing_train['BsmtQual'].fillna("No Basment")
missing_train['BsmtCond'] = missing_train['BsmtCond'].fillna("No Basment")
missing_train['BsmtExposure'] = missing_train['BsmtExposure'].fillna("No Basment")
missing_train['BsmtFinType1'] = missing_train['BsmtFinType1'].fillna("No Basment")
missing_train['BsmtFinType2'] = missing_train['BsmtFinType2'].fillna("No Basment")
missing_train['Electrical'] = missing_train['Electrical'].fillna(df['Electrical'].mode()[0])
missing_train['FireplaceQu'] = missing_train['FireplaceQu'].fillna("No Fireplace")
missing_train['GarageType'] = missing_train['GarageType'].fillna("No Garage")
missing_train['GarageFinish'] = missing_train['GarageFinish'].fillna("No Garage")
missing_train['GarageQual'] = missing_train['GarageQual'].fillna("No Garage")
missing_train['GarageCond'] = missing_train['GarageCond'].fillna("No Garage")
missing_train['Fence'] = missing_train['Fence'].fillna("No Fence")
missing_train['MiscFeature'] = missing_train['MiscFeature'].fillna("No Miscellaneous Feature")

missing_train.isna().sum()

#The function check_dtypes inspects the unique data types in each column of the dataframe.
def check_dtypes(missing_train):
    dtypes = []
    for col in missing_train.columns:
        dtypes.append( (col, missing_train[col].apply(type).unique()) )
    return pd.DataFrame(dtypes, columns=['Column Name', 'Data Type'])

check_dtypes(missing_train)

#*We check that the dataset does not have negative data

# Get the names of all columns with numeric values
numeric_columns = missing_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

#check negative value
negative_count = missing_train[numeric_columns].apply(lambda x: (x < 0).sum())

#Removing Outliers

missing_train.info()

missing_train.shape

missing_train.head()

missing_train.describe()

#Analysis and Visualization

df_num = train.select_dtypes(include = ['float64', 'int64'])
print(df_num.shape)
df_num.head()

missing_train['Street'].value_counts()

#Observation: The street Type of road access is pave according to df_train.

missing_train['LotShape'].value_counts()

#3 lotshape is Regular (Reg) according to df_train 3 lotshape is Slightly irregular (IR2) according to df_train 1 lotshape is Moderately irregular (IR3) according to df_train

missing_train['Utilities'].value_counts()

corr = df_num.drop(columns= 'SalePrice').corr()

fig , ax = plt.subplots(figsize=(25 , 20))

sns.heatmap(corr ,annot= True , ax=ax , cmap= 'Greens');

correlation_matrix=train[['MSSubClass','LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt','MiscVal', 'MoSold', 'YrSold', 'SalePrice']].corr()
sns.heatmap(correlation_matrix, annot=True)

# Create a figure with subplots
fig, axes = plt.subplots(5, 1, figsize=(8, 12))

# KDE plot for 'YrSold'
sns.kdeplot(data=df, x='YrSold', ax=axes[0], color='green', fill=True)
axes[0].set_title('Distribution of YrSold')

# KDE plot for 'SalePrice'
sns.kdeplot(data=df, x='SalePrice', ax=axes[1], color='blue', fill=True)
axes[1].set_title('Distribution of SalePrice')

# KDE plot for 'MoSold'
sns.kdeplot(data=df, x='MoSold', ax=axes[2], color='magenta', fill=True)
axes[2].set_title('Distribution of MoSold')

# KDE plot for 'OverallCond'
sns.kdeplot(data=df, x='OverallCond', ax=axes[3], color='red', fill=True)
axes[3].set_title('Distribution of Overall Condition')

# KDE plot for 'MSSubClass'
sns.kdeplot(data=df, x='MSSubClass', ax=axes[4], color='pink', fill=True)
axes[4].set_title('Distribution of MSSubClass')

# Adjust the layout and spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Select the columns of interest
cols = ['Street', 'Alley', 'LotShape', 'Utilities']

# Create the count plot subplot
sns.set_palette('pastel')
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, col in enumerate(cols):
    row = i // 2
    column = i % 2
    sns.countplot(x=col, data=train, ax=axs[row, column], palette='Set2')
    axs[row, column].set_title(col)
plt.tight_layout()
plt.show()

# Lets Make the Histogram of Salesprice column
sns.histplot(data=missing_train, x="SalePrice")
plt.xlabel('SalesPrice')
plt.ylabel('Count')
plt.title('Histogram of SalesPrice')

import math
def plot_numeric_column_with_price(ax, x):
    scatter = ax.scatter(x=df[x], y=df['SalePrice'], c=df['SalePrice'], cmap='viridis', alpha=0.8, s=10)
    ax.set(xlabel=x, ylabel='SalePrice', title=f'{x} With Sale Price')
    plt.colorbar(scatter, ax=ax, label='SalePrice')

columns = []
for c in df.select_dtypes('number').columns.tolist():
    if df[f'{c}'].nunique() > 16:
        columns.append(c)

num_rows = math.ceil(len(columns[:-1]) / 2)
fig, axes = plt.subplots(num_rows, 2, figsize=(18, 6 * num_rows))
axes = axes.flatten()
for i, column in enumerate(columns[:-1]):
    if i >= len(axes):
        break
    plot_numeric_column_with_price(axes[i], column)

plt.tight_layout()
plt.show()

def plot_categorical_column_with_price(x, ax):
    mean = df.groupby(x)['SalePrice'].mean()
    sns.barplot(x=mean.index, y=mean.values, width=0.5, ax=ax,palette='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel('SalePrice')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'Mean of Sale Price With {x}')

columns = df.select_dtypes(exclude='number').columns.tolist()
for c in df.select_dtypes('number').columns.tolist():
    if df[f'{c}'].nunique() <= 16:
        columns.append(c)

num_columns = 3
num_rows = (len(columns) + num_columns - 1) // num_columns

fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 6 * num_rows))

for i, column in enumerate(columns):
    row = i // num_columns
    col = i % num_columns
    if num_rows > 1:
        ax = axes[row, col]
    else:
        ax = axes[col]
    plot_categorical_column_with_price(column, ax)

for i in range(len(columns), num_rows * num_columns):
    row = i // num_columns
    col = i % num_columns
    if num_rows > 1:
        fig.delaxes(axes[row, col])
    else:
        fig.delaxes(axes[col])

plt.tight_layout()
plt.show()

imp_feat =train[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Neighborhood', 'LotArea',
'BsmtFinSF1', 'BsmtFinType1', '1stFlrSF', '2ndFlrSF', 'SalePrice']
]
imp_feat.head()

def plot_advanced_scatter(df, x, y, title=None):
    # Set background color and style
    sns.set_style("white", {"axes.facecolor": "#42EADDFF","figure.facecolor":"#42EADDFF"})  # Set background color
    sns.set_context("talk",font_scale = .7)
    plt.figure(figsize=(13, 5))
    sns.scatterplot(data=train, x=x, y=y, size=y, sizes=(20, 200),color='#4269DD')
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.legend(fontsize=10,title=y)
    plt.show()
    
plot_advanced_scatter(train , x='FullBath', y='SalePrice', title='Full Bath Vs SalePrice')

#Observation :

#Based on the provided data, it's evident that properties with a greater number of FullBath features command higher SalePrices. but when we see sale ratio the Houses with 2 Baths Have High Sale Ratio

plot_advanced_scatter(train , x='GarageCars', y='SalePrice', title='GarageCars Vs SalePrice')

#Observation :

#Based on the provided data, it's evident that properties with a GarageCars Of 3 command higher SalePrices.

import plotly.express as px
import pandas as pd
import numpy as np

# Group data by 'YearBuilt' and calculate mean 'BedroomAbvGr'
grouped_data = train.groupby('YearBuilt')['BedroomAbvGr'].mean().reset_index()

# Create trend plot
fig = px.line(grouped_data, x='YearBuilt', y='BedroomAbvGr',
              labels={'BedroomAbvGr': 'Mean BedroomAbvGr'},
              title='Trend of Mean BedroomAbvGr by Year Built',)
              
# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD",
                    plot_bgcolor='#42EADD')
                    
# Show plot
fig.show()

plot_advanced_scatter(train , x='BedroomAbvGr', y='SalePrice', title='BedroomAbvGr Vs SalePrice')

plot_advanced_scatter(train , x='OverallQual', y='SalePrice', title='OverallQual Vs SalePrice')

#Observation:

#According to the data provided, properties with an exceptional OverallQual rating of 10 tend to command higher SalePrices. but have low sale ratio

plot_advanced_scatter(train , 'OverallCond','SalePrice', title='OverallCond vs Sale')

#Observation:

#According to the data provided, properties with an Medium OverallCond rating of 6 tend to command higher SalePrices.

# Assuming 'YearBuilt' is the column for the year and 'SalePrice' is the column for sale prices
yearly_data = train[['YearBuilt', 'SalePrice']]
average_prices_by_year = yearly_data.groupby('YearBuilt')['SalePrice'].mean().reset_index()

# Create the plot
fig = px.line(average_prices_by_year, x='YearBuilt', y='SalePrice', title='Average SalePrice Over the Years')
fig.update_xaxes(title='Year Built')
fig.update_yaxes(title='Average SalePrice')

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD",
                    plot_bgcolor='#42EADD')
fig.show()

#Observations :

#It's notable from the above output that the average house price demonstrates a consistent increase on a year-by-year basis. This trend highlights a positive trajectory in the housing market.

# Grouping data by 'YearRemodAdd' and counting the occurrences
remodel_counts = train['YearRemodAdd'].value_counts().reset_index()
remodel_counts.columns = ['Year', 'Remodel Count']

# Sorting data by year
remodel_counts = remodel_counts.sort_values(by='Year')

# Plotting the trend over time
fig = px.line(remodel_counts, x='Year', y='Remodel Count', title='Trend of Property Remodeling Over Time')
fig.update_xaxes(title='Year of Remodeling')
fig.update_yaxes(title='Number of Remodeling Events')

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD",
                    plot_bgcolor='#42EADD')
fig.show()

#Observation :

#Based on the provided data, it's apparent that properties boasting four bedrooms (BedroomAbvGr = 4) tend to have higher SalePrices.but there a house with 8 bedrooms but the sale price is very low compatilby to other houses with less bedrooms . it becuase of the some factors with effect the house price

#Observation :

#Based on the data displayed above, it's clear that houses with a GrLivArea of 4316 and 4476 exhibit the highest sale price, albeit with a lower Sale Ratio. Conversely, properties with GrLivArea ranging from 630 to 2000 demonstrate the highest Sale Ratio. This suggests that despite the higher sale price for houses with a GrLivArea of 4316 and 4476, their sale ratio is comparatively lower compared to properties with GrLivArea within the 630 to 2000 range.

#Interestingly, certain properties boasting spacious GrLivArea measurements of 5642 and 4676 square feet exhibit lower sale prices compared to others with smaller living areas. Several factors may contribute to this price discrepancy. One potential explanation could be the condition or age of the properties, as older or poorly maintained homes may fetch lower prices despite their larger living spaces. Additionally, factors such as location, neighborhood amenities, layout efficiency, and the presence of desirable features like updated kitchens, bathrooms, or outdoor spaces can significantly influence property values. Furthermore, considering the GarageArea unit, properties with limited garage space relative to their living area may also affect their perceived value, particularly in areas where ample parking or storage space is highly valued.

# Grouping data by 'YearRemodAdd' and counting the occurrences
remodel_counts = train['YearRemodAdd'].value_counts().reset_index()
remodel_counts.columns = ['Year', 'Remodel Count']

# Sorting data by year
remodel_counts = remodel_counts.sort_values(by='Year')

# Plotting the trend over time
fig = px.line(remodel_counts, x='Year', y='Remodel Count', title='Trend of Property Remodeling Over Time')
fig.update_xaxes(title='Year of Remodeling')
fig.update_yaxes(title='Number of Remodeling Events')

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD",
                    plot_bgcolor='#42EADD')
fig.show()

#Observation :

#From the provided output, it's evident that there were 178 remodeling events in 1950. Subsequently, there was a notable decline in remodeling activity until around 1990. Following this period, there was a significant increase in the number of remodeling events, reaching a peak, but then experienced a sharp decrease to the lowest level observed in the dataset.

plot_advanced_scatter(train , x='TotalBsmtSF', y='SalePrice', title='TotalBsmtSF Vs SalePrice')

#Observation :

#From the provided output, it's evident that there were 178 remodeling events in 1950. Subsequently, there was a notable decline in remodeling activity until around 1990. Following this period, there was a significant increase in the number of remodeling events, reaching a peak, but then experienced a sharp decrease to the lowest level observed in the dataset.

plot_advanced_scatter(train , x='TotalBsmtSF', y='SalePrice', title='TotalBsmtSF Vs SalePrice')

#Observation :

#Based on the data displayed above, it's clear that houses with a TotalBsmtSF (Total Basement Square Footage) of 2444 exhibit the highest sale price, albeit with a lower Sale Ratio. Conversely, properties with TotalBsmtSF ranging from 480 to 1600 demonstrate the highest Sale Ratio. This suggests that despite the higher sale price for houses with a TotalBsmtSF of 2444, their sale ratio is comparatively lower compared to properties with TotalBsmtSF within the 480 to 1600 range.

#In the output above, we can observe that there are certain properties where TotalBsmtSF equals zero. These instances are considered outliers because they deviate from the typical pattern or distribution observed in the dataset.

#Interestingly, certain properties boasting spacious TotalBsmtSF measurements of 6110 and 3138 square feet exhibit lower sale prices compared to others with smaller living areas. Several factors may contribute to this price discrepancy. One potential explanation could be the condition or age of the properties, as older or poorly maintained homes may fetch lower prices despite their larger living spaces. Additionally, factors such as location, neighborhood amenities, layout efficiency, and the presence of desirable features like updated kitchens, bathrooms, or outdoor spaces can significantly influence property values. Furthermore, considering the GarageArea unit, properties with limited garage space relative to their living area may also affect their perceived value, particularly in areas where ample parking or storage space is highly valued.

plot_advanced_scatter(train , x='GrLivArea', y='SalePrice', title='GrLivArea Vs SalePrice')

#Observation :

#Based on the data displayed above, it's clear that houses with a GrLivArea of 4316 and 4476 exhibit the highest sale price, albeit with a lower Sale Ratio. Conversely, properties with GrLivArea ranging from 630 to 2000 demonstrate the highest Sale Ratio. This suggests that despite the higher sale price for houses with a GrLivArea of 4316 and 4476, their sale ratio is comparatively lower compared to properties with GrLivArea within the 630 to 2000 range.

#Interestingly, certain properties boasting spacious GrLivArea measurements of 5642 and 4676 square feet exhibit lower sale prices compared to others with smaller living areas. Several factors may contribute to this price discrepancy. One potential explanation could be the condition or age of the properties, as older or poorly maintained homes may fetch lower prices despite their larger living spaces. Additionally, factors such as location, neighborhood amenities, layout efficiency, and the presence of desirable features like updated kitchens, bathrooms, or outdoor spaces can significantly influence property values. Furthermore, considering the GarageArea unit, properties with limited garage space relative to their living area may also affect their perceived value, particularly in areas where ample parking or storage space is highly valued.

plot_advanced_scatter(train , x='KitchenAbvGr', y= 'SalePrice' , title='KitchenAbvGr Vs SalePrice')

#Observation :

#Based on the data shown, homes with one kitchen located above ground tend to have the highest prices and also boast a higher sale ratio. This suggests that such properties are in greater demand and command premium prices in the market.

plot_advanced_scatter(train , x='TotRmsAbvGrd', y='SalePrice', title='TotRmsAbvGrd Vs SalePrice')

#Observation :

#Based on the provided output, homes with a Total Rooms Above Grade (TotRmsAbvGrd) value of 10 exhibit the highest SalePrice. Additionally, residences with a TotRmsAbvGrd value of 6 have the highest count among the dataset, accompanied by an average SalePrice.

plot_advanced_scatter(train , x='GarageArea', y='SalePrice', title='GarageArea Vs SalePrice')

#Observation :

#Based on the data presented above, it is evident that houses with a GarageArea of 832 and 813 exhibit the highest sale prices, albeit with a lower Sale Ratio. Conversely, properties with GarageArea ranging from 200 to 600 square feet demonstrate the highest Sale Ratio. This suggests that despite the higher sale price for houses with a GarageArea of 832 and 813, their sale ratio is comparatively lower compared to properties with GarageArea within the 200 to 600 square feet range.

#There are exceptions observed among properties with GarageArea values of 1248, 1356, 1390, and 1418, which have notably lower SalePrices. This deviation could be attributed to factors such as the condition of the garage, its functionality, or other specific features that may not align with typical market expectations, ultimately affecting the overall valuation of these properties.

#In the output above, we can observe that there are certain properties where GarageArea equals zero. These instances are considered outliers because they deviate from the typical pattern or distribution observed in the dataset.

# plot_scatter(df_train, x='LotArea', y='SalePrice', title='LotArea Vs SalePrice')
plot_advanced_scatter(train , x='LotArea', y='SalePrice', title='LotArea Vs SalePrice')

#Observations :

#Based on the provided data, it's observed that properties with LotArea of 21,535 square feet and 15,623 square feet command the highest SalePrices, yet they exhibit relatively low sale ratios. Conversely, properties with LotArea ranging from 70,000 to 115,000 square feet demonstrate higher sale ratios and boast SalePrices above the average. Notably, certain properties with larger LotArea sizes have comparatively lower SalePrices. This could be attributed to factors such as location desirability, property condition, or amenities offered, which may not align with the increased LotArea.

# plot_scatter(df_train, x='BsmtFinSF1', y='SalePrice', title='BsmtFinSF1 Vs SalePrice')
plot_advanced_scatter(train , x='BsmtFinSF1', y='SalePrice', title='BsmtFinSF1 Vs SalePrice')

#Observations:

#Upon analyzing the provided data, it is noted that properties with BsmtFinSF1 ranging from 1455 to 2096 square feet command the highest SalePrices. However, despite their larger sizes, they exhibit relatively low sale ratios. Conversely, properties with BsmtFinSF1 ranging from 100 to 1100 square feet demonstrate higher sale ratios and achieve SalePrices close to the average. Notably, certain properties with larger BsmtFinSF1 sizes have comparatively lower SalePrices. This phenomenon could be attributed to factors such as location desirability, property condition, or amenities offered, which may not align with the increased BsmtFinSF1 size.

#In the output above, we can observe that there are certain properties where GarageArea equals zero. These instances are considered outliers because they deviate from the typical pattern or distribution observed in the dataset.

plot_advanced_scatter(train , x='1stFlrSF', y='SalePrice', title='1stFlrSF Vs SalePrice')
plot_advanced_scatter(train, x='2ndFlrSF', y='SalePrice', title='2ndFlrSF Vs SalePrice')

#Observation:

#Upon reviewing the provided data, it's observed that properties with a 1stFlrSF (First Floor Square Footage) ranging between 2411 and 2444 square feet command the highest SalePrices. However, despite their larger sizes, they exhibit relatively low sale ratios. Conversely, properties with a 1stFlrSF ranging from 500 to 1800 square feet demonstrate higher sale ratios and achieve SalePrices close to or above the average. Notably, certain properties with larger 1stFlrSF sizes have comparatively lower SalePrices. This phenomenon could be attributed to factors such as location desirability, property condition, or amenities offered, which may not align with the increased BsmtFinSF1 size.

#Upon reviewing the provided data, it's observed that properties with a 2ndFlrSF (Second Floor Square Footage) ranging between 1872 and 2065 square feet command the highest SalePrices. However, despite their larger sizes, they exhibit relatively low sale ratios. Conversely, properties with a 2ndFlrSF ranging from 500 to 1000 square feet demonstrate higher sale ratios and achieve SalePrices close to the average SalePrice.

#In the output above, we can observe that there are certain properties where GarageArea equals zero. These instances are considered outliers because they deviate from the typical pattern or distribution observed in the dataset.

def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules.
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    
    ## Set the title.
    ax1.set_title('Histogram')
    ## plot the histogram.
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title.
    ax2.set_title('QQ_plot')
        ## Plotting the QQ_Plot.
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title.
    ax3.set_title('Box Plot')
    ## Plotting the box plot.
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

plotting_3_chart(train, 'SalePrice')

#missing_train

#missing_train1=missing_train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

#saleprice correlation matrix
k = 15 #number of variables for heatmap
plt.figure(figsize=(16,8))
corrmat = missing_train1.corr()

# picking the top 15 correlated features
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(missing_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# making plot with sale price to other feat]ures scatter and histogram
figsize = (10, 5)
sns.scatterplot(x='SalePrice', y='OverallQual', data=missing_train)
plt.xlabel('Sale Price')
plt.ylabel('Overall Quality')
plt.title('Scatter Plot: Sale Price vs Overall Quality')
plt.show()

# Histogram
plt.figure(figsize=figsize)
sns.histplot(missing_train['SalePrice'])
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.title('Histogram: Sale Price Distribution')
plt.show()

# Data Visualization
# Visualize the distribution of the target variable 'SalePrice'

sns.histplot(missing_train['SalePrice'], kde=True, color=(0.2, 0.5, 0.1))
plt.title('Distribution of SalePrice')
plt.show()

# Extract numerical columns
numerical_columns = missing_train.select_dtypes(include=['int64', 'float64']).columns

# Calculate the number of rows needed
num_rows = (len(numerical_columns) + 2) // 3

# Set up subplots with 3 columns
fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
fig.subplots_adjust(hspace=0.5)

# Plot KDE for each numerical column
for i, column in enumerate(numerical_columns):
    row_idx, col_idx = divmod(i, 3)
    sns.kdeplot(data=missing_train, x=column, ax=axes[row_idx, col_idx], fill=True)
    axes[row_idx, col_idx].set_title(f'KDE Plot for {column}')

# Remove empty subplots if the number of plots is not a multiple of 3
for i in range(len(numerical_columns), num_rows * 3):
    fig.delaxes(axes.flatten()[i])

plt.show()

#The data is not normally distributed, many features are skewed and other features exhibits different bimodal and multimodal peaks and kurtosis

# Select the numerical columns for visualization
numerical_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                      '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'GarageYrBlt',
                      'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                      'MiscVal', 'YrSold', 'SalePrice']

# Normalize the selected columns
normalized_data = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()

# Set a stylish color palette
sns.set_palette("husl")

# Plot KDE plots
plt.figure(figsize=(15, 20))

# Loop through each column and plot KDE
for column in normalized_data.columns:
    sns.kdeplot(normalized_data[column], label=column, linewidth=2, shade=True)

#Insights:

#This KDE plots provide a clear overview of the normalized distribution for each numerical feature in the House Price dataset. The normalization process ensures that the features are on a consistent scale, aiding in the identification of patterns and potential outliers. Observing the shapes and peaks of the KDEs can offer insights into the underlying data distribution, facilitating a better understanding of the dataset's characteristics. The visualization can be particularly useful for assessing the relative importance of different features and detecting any potential skewness or multimodality in their distributions.

#SalePrice Evaluation with time using Ploty and Seaborn+Matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame containing the data
# Make sure 'SalePrice', 'YearBuilt', and 'YearRemodAdd' are columns in df

# Set the style for Seaborn
sns.set(style="whitegrid")

# Create a line plot using Seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=missing_train, x='YearBuilt', y='SalePrice', label='YearBuilt', ci=None)
sns.lineplot(data=missing_train, x='YearRemodAdd', y='SalePrice', label='YearRemodAdd', ci=None)

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Sale Price')
plt.title('Sale Price over Years')

# Show legend
plt.legend()

# Show the plot
plt.show()

#This code uses Seaborn and Matplotlib to create a line plot with YearBuilt (year built) and YearRemodAdd (year remodification) on the x-axis, SalePrice on the y-axis, and different lines representing Sale Prices over the years.

from sklearn.preprocessing import MinMaxScaler

# Specify the columns to scale
columns_to_scale = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                     'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                     'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                     'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

# Create a MinMaxScaler with the specified range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

from sklearn.preprocessing import MinMaxScaler

# Sample DataFrame (replace this with your actual DataFrame)
# df = ...

# Specify the columns to scale
columns_to_scale = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                     'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                     'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                     'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

# Create a MinMaxScaler with the specified range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Plotting the histograms before and after scaling
plt.figure(figsize=(15, 8))

# Before Scaling
plt.subplot(1, 2, 1)
sns.set(style="whitegrid")
sns.histplot(data=df[columns_to_scale].drop(columns=['SalePrice']), kde=True)
plt.title("Before Scaling")

# After Scaling
plt.subplot(1, 2, 2)
sns.set(style="whitegrid")
sns.histplot(data=df[columns_to_scale].drop(columns=['SalePrice']), kde=True)
plt.title("After Scaling")

plt.tight_layout()
plt.show()

#Lately We would like to use the Random Forest Regressor algorithm, scaling features is generally less critical due to the algorithm's non-parametric and ensemble nature. Random Forest is insensitive to the absolute scale of features, and its decision-making process is based on the relative ordering of values. While scaling might not have a significant impact on the algorithm's performance, it remains a good practice for consistent behavior across different algorithms and in cases where features have disparate scales, ensuring better convergence during tree-building.

#normal distribution\
from scipy import stats
from scipy.stats import norm, skew
sns.distplot(missing_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(missing_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#Log Transformation

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
missing_train["SalePrice"] = np.log1p(missing_train["SalePrice"])

#Check the new distribution
sns.distplot(missing_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(missing_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(missing_train['SalePrice'], plot=plt)
plt.show()

#Why we use log tranformer?

#Skewed Data: Reduces the impact of extreme values in positively skewed data. Variance Stabilization: Stabilizes the variance across different levels of an independent variable. Normalization: Makes data more normally distributed, which is often assumed in statistical models. Interpretability: Provides a more interpretable scale, especially for percentage changes. Equalizing Effects: Ensures equal importance of errors in predicting high and low values. Multiplicative Relationships: Converts multiplicative relationships into additive ones. Homoscedasticity: Helps address unequal spread of residuals in regression models. bold text

#Visualizations Of Outlier

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
num_1_out = ['Id','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
 'BsmtUnfSF']

# Calculate the number of rows and columns needed for subplots
num_cols = 2
num_rows = (len(num_1_out) + 1) // num_cols

# Create subplots with the calculated number of rows and columns
fig = make_subplots(rows=num_rows, cols=num_cols)

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD",
                    plot_bgcolor='#42EADD')

# Loop through numerical columns and add boxplots with color
for i, col in enumerate(num_1_out, start=1):
    row_num = (i - 1) // num_cols + 1
    col_num = (i - 1) % num_cols + 1
    fig.add_trace(
           go.Box(
            x=train[col],
            name=col,
            marker_color='#1f77b4',  # Set box color
            line_color='#1f77b4'     # Set mean line color
        ),
        row=row_num,
        col=col_num
    )

# Update layout
fig.update_layout(
    title_text="Boxplots of Numerical Columns",
    showlegend=False
)
# Show the plot
fig.show()

# List of numerical columns for outliers
num_2_out = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
             'GarageYrBlt']

# Calculate the number of rows and columns needed for subplots
num_cols = 2
num_rows = (len(num_2_out) + num_cols - 1) // num_cols  # Calculate number of rows based on columns

# Create subplots with the calculated number of rows and columns
fig = make_subplots(rows=num_rows, cols=num_cols)

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')

# Loop through numerical columns and add boxplots with color
for i, col in enumerate(num_2_out, start=1):
    row_num = (i - 1) % num_rows + 1
    col_num = (i - 1) // num_rows + 1
    fig.add_trace(
        go.Box(
            x=train[col],
             name=col,
            marker_color='#1f77b4',  # Set box color
            line_color='#1f77b4'     # Set mean line color
        ),
        row=row_num,
        col=col_num
    )

# Update layout
fig.update_layout(title_text="Boxplots of Numerical Columns", showlegend=False)

# Show the plot
fig.show()

num_3_out =  ['GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'MoSold',
 'YrSold',
 'SalePrice']

# Calculate the number of rows and columns needed for subplots
num_cols = 2
num_rows = (len(num_3_out) + num_cols - 1) // num_cols  # Calculate number of rows based on columns

# Create subplots with the calculated number of rows and columns
fig = make_subplots(rows=num_rows, cols=num_cols)

# Update layout with palette color as background
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')

# Loop through numerical columns and add boxplots with color
for i, col in enumerate(num_3_out, start=1):
    row_num = (i - 1) % num_rows + 1
    col_num = (i - 1) // num_rows + 1
    fig.add_trace(
        go.Box(
            x=train[col],
            name=col,
            marker_color='#1f77b4',  # Set box color
            line_color='#1f77b4'     # Set mean line color
        ),
        row=row_num,
        col=col_num
    )

# Update layout
fig.update_layout(title_text="Boxplots of Numerical Columns", showlegend=False)

# Show the plot
fig.show()

#show the Distribution of SalePrice/Electrical Ratio: x = Electrical , y = SalePrice
sns.boxenplot(x='Electrical' , y= 'SalePrice' , data=train)
plt.xlabel("Electrical")
plt.ylabel("SalePrice")
plt.title("Distribution of SalePrice/Electrical Ratio: x = ");

def remove_outliers_zscore(missing_train, columns, threshold=4):
    for col in columns:
        # Calculate the mean and standard deviation
        mean = missing_train[col].mean()
        std_dev = missing_train[col].std()

        # Calculate the Z-score for each value in the column
        z_scores = (missing_train[col] - mean) / std_dev

        # Filter the DataFrame to show outlier values and calculate the sum
        outliers_sum = missing_train.loc[abs(z_scores) > threshold, col].value_counts().sum()
        print(f"Total number of outliers detected and Deleted are : {outliers_sum}")

        # Remove outliers from the dataset
        df_clean = missing_train.loc[abs(z_scores) <= threshold].copy()

    return df_clean

#Plot Box Plot to Seee Outliers in LotArea in Plotly
fig = px.box(missing_train, y="LotArea")
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')
fig.show()

missing_train = remove_outliers_zscore(missing_train, ['LotArea'])

#Plot Box Plot to Seee Outliers in LotArea in Plotly
fig = px.box(missing_train, y="GrLivArea")
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')
fig.show()

missing_train= remove_outliers_zscore(missing_train,['GrLivArea'])

#Plot Box Plot to Seee Outliers in LotArea in Plotly
fig = px.box(missing_train, y="LotFrontage")
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')
fig.show()

missing_train = remove_outliers_zscore(missing_train, ['LotFrontage'])

#Plot Box Plot to Seee Outliers in LotArea in Plotly
fig = px.box(missing_train, y="MasVnrArea")
fig.update_layout(paper_bgcolor="#42EADD", plot_bgcolor='#42EADD')
fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Numerical columns
numerical_columns = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                     'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

# Create subplots with 2 columns
fig = make_subplots(rows=len(numerical_columns)//2, cols=2, subplot_titles=numerical_columns)

# Loop through each numerical column and add its boxplot to the subplot
for i, column in enumerate(numerical_columns):
    row = i // 2 + 1  # Calculate row number for subplot
    col = i % 2 + 1   # Calculate column number for subplot
    fig.add_trace(go.Box(y=df[column], name=column), row=row, col=col)

# Update layout
fig.update_layout(height=750, width=1200, title_text="Boxplots of Numerical Columns")

# Show plot
fig.show()

outlier_percentage = {}

for col in numerical_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_percentage[col] = len(outliers) / len(df) * 100

print("Outlier Percentage:")
for col, percentage in outlier_percentage.items():
    print(f"{col}: {percentage:.2f}%")

missing_train = remove_outliers_zscore(missing_train, ['MasVnrArea'])

def outliner_detector(missing_train, cols, take_care_outliners=False, print_outliners=False, q_1=0.25, q_3=0.75):

    temp = pd.DataFrame()
    data = missing_train.copy()

    for col in cols:

        q1 = data[col].quantile(q_1)
        q3 = data[col].quantile(q_3)
        IQR = q3 - q1
        up = q3 + 1.5 * IQR
        low = q1 - 1.5 * IQR
        temp.loc[col, "Min"] = round(data[col].min())
        temp.loc[col, "Low_Limit"] = round(low)
        temp.loc[col, "Mean"] = round(data[col].mean())
        temp.loc[col, "Median"] = round(data[col].median())
        temp.loc[col,"Up_Limit"] = up
        temp.loc[col, "Max"] = data[col].max()
        temp.loc[col, "Outliner"] = "Min-Max-Outliner" if (data[col].max() > up) & (low > data[col].min())\
                                    else ("Max-Outliner" if data[col].max() > up \
                                    else ("Min-Outliner" if low > data[col].min() \
                                    else "No"))
        if take_care_outliners:
            data.loc[data[col] > up,col] = round(up-1)
            data.loc[data[col] < low,col] = round(low-1)
    if take_care_outliners:
        if print_outliners: return temp
        return data
    if print_outliners: return temp

def mice_imput(df:pd.DataFrame, fill:str, based:list) -> pd.Series :
    """
    Impute missing values in a specified column of a DataFrame using the
    MICE (Multiple Imputation by Chained Equations) algorithm.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - fill (str): The column name with missing values to be imputed.
    - based (list): A list of column names considered as features for imputation.

    Returns:
    - pd.Series: A Series containing the imputed values for the specified column.

   # MICE (Multiple Imputation by Chained Equations) is a statistical method used for imputing
   # missing data in a dataset.
    #It is an iterative algorithm that imputes missing values one variable at a time,
   # considering the relationships between variables. In this implementation:

   # 1. Categorical columns are identified in the 'based' list.
    #2. A temporary DataFrame is created by one-hot encoding categorical columns and
    #selecting the target column ('fill').
    #3. A missing value mask is generated for the temporary DataFrame.
    #4. The IterativeImputer from scikit-learn is used to impute missing values iteratively.
    5. The imputed values are assigned to the original DataFrame in the specified column.
    """

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    categoric_cols = [col for col in based if df[col].dtype == "O"]

    temp_df = pd.get_dummies(df[[fill] + based].copy(), columns=categoric_cols)

    missing_mask = temp_df.isna()

    imputer = IterativeImputer(max_iter=10, random_state=42)

    imputed_values = imputer.fit_transform(temp_df)

    temp_df[fill][temp_df[fill].isnull()] = imputed_values[missing_mask]

    return temp_df[fill]

missing_train["LotFrontage"] = mice_imput(missing_train, fill="LotFrontage", based=["LotArea","LotShape","LotConfig"])
missing_train.loc[missing_train["MasVnrArea"].isnull(),["MasVnrArea"]] = 0
missing_train[cat_cols] = missing_train[cat_cols].fillna("None")

df["Street"].value_counts() / len(df) * 100

df["HeatingQC"].value_counts() / len(df) * 100

train.columns = [col.replace(" ", "_") for col in train.columns]
test.columns = [col.replace(" ", "_") for col in test.columns]

#Now, let's ep dive into normalization techniques. After removing the outliers and setting a threshold of 4, we observe a slight improvement in numerical skewness. Now, let's harness the power of normalization techniques like Box-Cox, Yeo-Johnson, and Quantile Transformation.

imp_feat = train[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Neighborhood', 'LotArea',
'BsmtFinSF1', 'BsmtFinType1', '1stFlrSF', '2ndFlrSF']
]
imp_feat.head()

#Preprocessing

#Ecoding the data

missing_train = pd.get_dummies(missing_train,dtype=float,drop_first=True)

#Split the data

missing_train.head()

#Creating New Feature

numerical_df['Age_House']= (numerical_df['YrSold']-numerical_df['YearBuilt'])
numerical_df['Age_House'].describe()

Negatif = numerical_df[numerical_df['Age_House'] < 0]
Negatif

numerical_df.loc[numerical_df['YrSold'] < numerical_df['YearBuilt'],'YrSold' ] = 2009
numerical_df['Age_House']= (numerical_df['YrSold']-numerical_df['YearBuilt'])
numerical_df['Age_House'].describe()

numerical_df['TotalBsmtBath'] = numerical_df['BsmtFullBath'] + numerical_df['BsmtFullBath']*0.5
numerical_df['TotalBath'] = numerical_df['FullBath'] + numerical_df['HalfBath']*0.5
numerical_df['TotalSA']=numerical_df['TotalBsmtSF'] + numerical_df['1stFlrSF'] + numerical_df['2ndFlrSF']

numerical_df.head()

missing_train=missing_train.dropna()

missing_train.isna().sum()

# Renaming The DataFrames Again BAck to Train and Test
df_train = missing_train.copy()
df_test = missing_test.copy()

print(f'The Shape Of Train Data After Adding New Features is : {df_train.shape}')
print(f'The Shape Of Test Data After Adding New Features is : {df_test.shape}')

# Selecting only numerical columns from the DataFrame
numerical_features = df_train.select_dtypes(include=['int64', 'float64'])

# Calculating correlation matrix
correlation_matrix = numerical_features.corr()

# Filtering features with correlation >= 0.5 with the target variable
high_corr_features = correlation_matrix[correlation_matrix['SalePrice'] >= 0.5]

# Returning high correlated features in table form
high_corr_features_table = pd.DataFrame(high_corr_features['SalePrice'])

# Renaming the column
high_corr_features_table.columns = ['SalePrice']

print(high_corr_features_table)

#Machine Learning

df_train = pd.get_dummies(df_train,dtype=float,drop_first=True)

df_train=df_train.drop(['1stFlrSF','2ndFlrSF','BsmtFullBath','FullBath','GarageYrBlt','GarageArea','GarageArea','Alley_Pave','PoolQC_Fa','PoolQC_Gd','PoolQC_None','Fence_GdWo','Fence_MnPrv','Fence_MnWw','Fence_No Fence','MiscFeature_No Miscellaneous Feature','MiscFeature_Othr','MiscFeature_Shed','MiscFeature_TenC','Alley_No Alley Acess','TotRmsAbvGrd','train'],axis=1)

df_train=df_train.drop(['MSZoning', 'KitchenQual','Street','Functional','LotShape','FireplaceQu','LandContour','Utilities','LotConfig','Neighborhood','LandSlope','Condition1','Condition2','BldgType','GarageType','HouseStyle','OverallQual', 'OverallCond','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','WoodDeckSF','SaleType','SaleCondition'],axis=1)

df_train

X = df_train.drop(['SalePrice'],axis = 1)
y = df_train['SalePrice']

scaler = MinMaxScaler()
encoder = LabelEncoder()

for col in X.columns:
    if X[col].dtype != 'object':
        X[col] = scaler.fit_transform(X[[col]])
    if X[col].dtype == 'object':
        X[col] = encoder.fit_transform(X[col])

missing_train.isna().sum()

pca = PCA(whiten=True)
pca.fit(df_train)
variance = pd.DataFrame(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train , X_test , y_train , y_test = train_test_split(X ,y ,test_size=0.2 , random_state=42 )
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
X_train shape: (1080, 27)
y_train shape: (1080,)
X_test shape: (271, 27)
y_test shape: (271,)

#Build Baseline

from sklearn.metrics import mean_absolute_error
y_mean = y_train.mean()
y_pred_baseline= [y_mean] * len(y_train)
print("Mean apt price:", y_mean)

print("Baseline MAE:", mean_absolute_error(y_train,y_pred_baseline))

# Import Libararies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Import Standaradscalar, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
#import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Import train test split
from sklearn.model_selection import train_test_split

# for Mchine learning
#For Regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

# import pipeline
from sklearn.pipeline import Pipeline

# import GridSearchCv
from sklearn.model_selection import GridSearchCV

# import cross validation score
from sklearn.model_selection import cross_val_score

# import Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import Deep Learning Libararies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import Earlystopping
from tensorflow.keras.callbacks import EarlyStopping

# Remove Warnings
import warnings
warnings.filterwarnings("ignore")

# Define the models to evaluate
models = {
    'Ridge': (Ridge(), {'alpha': [0.1, 1, 10]}),
    'LinearRegression':(LinearRegression(), {}),
    'DecisionTreeRegressor':(DecisionTreeRegressor(),{'criterion':['squared_error','absolute_error']}),
    'RandomForestRegressor':(RandomForestRegressor(),{'n_estimators':[10,50], 'max_depth':[3,4]}),
    'KNeighborsRegressor':(KNeighborsRegressor(),{'n_neighbors':np.arange(2,5,9),'weights':['uniform']}),
    'SVR':(SVR(), {'kernel':['rbf','poly','sigmoid'],'C':[0.1,1],'gamma':[0.1,0.01]}),
    'XGBRegressor':(XGBRegressor(),{'n_estimators':[10,50],'loss': ['ls', 'lad', 'huber', 'quantile']}),
    'GradientBoostingRegressor':(GradientBoostingRegressor(),{'n_estimators': [10,50],'learning_rate': [0.1, 0.01, 0.001]})
}

best_model = None
best_rmse = float('inf')

for name, (model, param_grid) in models.items():
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=2)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Make predictions
    y_pred = grid_search.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Check if this model has the lowest RMSE so far
    if rmse < best_rmse:
        best_model = grid_search.best_estimator_
        best_rmse = rmse

    # Print the evaluation metrics
    print(f"Model: {name}")
    print('Root_mean_squared_error:', rmse)
    print(name, 'Best parameters:', grid_search.best_params_)
    print('\n')

# Print the best model
print("Best Model:", best_model)

import matplotlib.pyplot as plt
import seaborn as sns

# Model performance data
models = [
    "Ridge", "LinearRegression", "DecisionTreeRegressor",
    "RandomForestRegressor", "KNeighborsRegressor",
    "SVR", "XGBRegressor", "GradientBoostingRegressor"
]

mean_absolute_errors = [0.15903141779788518, 0.1589344575758852, 0.07264957264957264,
                        0.22468803418803415, 0.03548290598290599, 0.2138738143083785,
                        0.046153846153846156, 0.18948350854341184]

mean_squared_errors = [0.025564695496173982, 0.02554743257509199, 0.006628679962013296,
                       0.05077409060851777, 0.0014823730367448324, 0.04661668094011337,
                       0.0040389442148834115, 0.036929475898056376]

root_mean_squared_errors = [0.15988963536193954, 0.1598356423802025, 0.08141670566912725,
                            0.22533106889312396, 0.038501597846645697, 0.21590896447371835,
                            0.06355268849453508, 0.1921704345055617]
combined_scores = [0.3444857486559987, 0.3443175325311797, 0.16069495828071317,
                   0.5007931936896759, 0.07546687686629652, 0.4763994597222102,
                   0.11374547886326464, 0.4185834189470299]

best_parameters = [
    {'alpha': 0.1}, {}, {'criterion': 'squared_error'},
    {'max_depth': 3, 'n_estimators': 10}, {'n_neighbors': 2, 'weights': 'uniform'},
    {'C': 0.1, 'gamma': 0.1, 'kernel': 'poly'}, {'loss': 'ls', 'n_estimators': 10},
    {'learning_rate': 0.1, 'n_estimators': 10}
]

# Unique plots
plt.figure(figsize=(13, 9))

# Plotting mean absolute error vs. model
plt.subplot(2, 2, 1)
sns.barplot(x=models, y=mean_absolute_errors, palette='muted')
plt.title('Mean Absolute Error for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)

# Plotting mean squared error vs. model
plt.subplot(2, 2, 2)
sns.barplot(x=models, y=mean_squared_errors, palette='viridis')
plt.title('Mean Squared Error for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)

# Plotting root mean squared error vs. model
plt.subplot(2, 2, 3)
sns.barplot(x=models, y=root_mean_squared_errors, palette='dark')
plt.title('Root Mean Squared Error for Different Models')
plt.xlabel('Model')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=45)

# Plotting combined scores vs. model
plt.subplot(2, 2, 4)
sns.barplot(x=models, y=combined_scores, palette='bright')
plt.title('Combined Scores for Different Models')
plt.xlabel('Model')
plt.ylabel('Combined Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Observations: I used Ridge Regression, LinearRegression, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor, SVR, XGBRegressor, GradientBoostingRegressor Models and Undergoes the Hyperparameter Tuning and choose The best Model from all of them using GridSearchCV. I found that the Best Model based on RMSE is Model: SVR
Root_mean_squared_error: 0.14783289540818184
SVR Best parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}

#LinearRegression

from sklearn.linear_model import LinearRegression
Lrg = LinearRegression()

Lrg.fit(X_train , y_train);

mean_absolute_error(y_train , Lrg.predict(X_train))

reg_y_pred_train = Lrg.predict(X_train)

reg_y_pred_test = Lrg.predict(X_test)

reg_acc_train = r2_score(y_train , reg_y_pred_train)
reg_acc_test = r2_score(y_test , reg_y_pred_test)

print("Training Accuracy:", round(reg_acc_train, 4))
print("Test Accuracy:", round(reg_acc_test, 4))

#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)

#Evaluation the model

y_pred = poly_lr.predict(X_test_poly)
y_train_pred = poly_lr.predict(X_train_poly)
r2_poly_train = r2_score(y_train,y_train_pred)
r2_poly_test = r2_score(y_test,y_pred)
print("R2 Train Score:", r2_poly_train)
print("R2 Test Score:", r2_poly_test)

#Ridge classiffier

rge = Ridge()
rge.fit(X_train , y_train);

mean_absolute_error(y_train , rge.predict(X_train))

rge_y_pred_test = rge.predict(X_test)
rge_y_pred_train = rge.predict(X_train)

rge_acc_train = r2_score(y_train , rge_y_pred_train)
rge_acc_test = r2_score(y_test , rge_y_pred_test)

print("Training Accuracy:", round(rge_acc_train, 4))
print("Test Accuracy:", round(rge_acc_test, 4))

#Robust Scaler

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
lso =  make_pipeline(RobustScaler(), Lasso(alpha =0.0005))

lso.fit(X_train , y_train);

mean_absolute_error(y_train , lso.predict(X_train))

lso_y_pred_test = lso.predict(X_test)
lso_y_pred_train = lso.predict(X_train)

lso_acc_train = r2_score(y_train , lso_y_pred_train)
lso_acc_test = r2_score(y_test , lso_y_pred_test)

print("Training Accuracy:", round(lso_acc_train, 4))
print("Test Accuracy:", round(lso_acc_test, 4))

#** RandomForestRegressor**

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          random_state=42)

rf.fit(X_train , y_train);

print (rf.score(X_train , y_train))
print (rf.score(X_test , y_test))

mean_absolute_error(y_train , rf.predict(X_train))

rf_y_pred_test =rf.predict(X_test)
rf_y_pred_train = rf.predict(X_train)

rf_acc_train = r2_score(y_train , rf_y_pred_train)
rf_acc_test = r2_score(y_test , rf_y_pred_test)

print("Training Accuracy:", round(rf_acc_train, 4))
print("Test Accuracy:", round(rf_acc_test, 4))

#** DecisionTreeRegressor**

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          random_state=42)
dt.fit(X_train , y_train);

print (dt.score(X_train , y_train))
print (dt.score(X_test , y_test))

mean_absolute_error(y_train , dt.predict(X_train))

dt_y_pred_test =dt.predict(X_test)
dt_y_pred_train =dt.predict(X_train)

dt_acc_train = r2_score(y_train , dt_y_pred_train)
dt_acc_test = r2_score(y_test , dt_y_pred_test)

print("Training Accuracy:", round(dt_acc_train, 4))
print("Test Accuracy:", round(dt_acc_test, 4))

#** XGBRegressor**

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators= 1000 , max_depth= 3 , learning_rate = 0.01)
xgb.fit(X_train , y_train);

print (xgb.score(X_train , y_train))
print (xgb.score(X_test , y_test))

mean_absolute_error(y_train , xgb.predict(X_train))

xgb_y_pred_test =xgb.predict(X_test)
Xgb_y_pred_train =xgb.predict(X_train)

xgb_acc_train = r2_score(y_train , Xgb_y_pred_train)
xgb_acc_test = r2_score(y_test , xgb_y_pred_test)

print("Training Accuracy:", round(xgb_acc_train, 4))
print("Test Accuracy:", round(xgb_acc_test, 4))

#CatBoostRegressor

pip install catboost

from catboost import CatBoostClassifier
from sklearn.metrics import mean_absolute_percentage_error
# Base model
model = CatBoostClassifier(silent=True)
model.fit(X_train,y_train)
Y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, Y_pred)
mape = mean_absolute_percentage_error(y_test, Y_pred)

print(f"r2: {r2}")
print(f"mean absolute percentage error: {mape}")

model_y_pred_test =model.predict(X_test)
model_y_pred_train =model.predict(X_train)

model_acc_train = r2_score(y_train , model_y_pred_train)
model_acc_test = r2_score(y_test , model_y_pred_test)

print("Training Accuracy:", round(model_acc_train, 4))
print("Test Accuracy:", round(model_acc_test, 4))

#Linear SVR

svr = SVR(kernel = 'linear')
svr.fit(X_train, y_train)

#Evaluation the model

y_pred = svr.predict(X_test)
y_train_pred = svr.predict(X_train)
r2_svr_lr_train = r2_score(y_train, y_train_pred)
r2_svr_lr_test = r2_score(y_test, y_pred)
print("R2 Train Score:", r2_svr_lr_train)
print("R2 Test Score:", r2_svr_lr_test)
mse_svr_lr_train = mean_squared_error(y_train, y_train_pred)
mse_svr_lr_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error of Train:", mse_svr_lr_train)
print("Mean Squared Error of Test:", mse_svr_lr_test)

#Lasso

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

#Evaluation the model

y_pred = lasso.predict(X_test)
y_train_pred = lasso.predict(X_train)
r2_lasso_train = r2_score(y_train, y_train_pred)
r2_lasso_test = r2_score(y_test, y_pred)
print("R2 Train Score:", r2_lasso_train)
print("R2 Test Score:", r2_lasso_test)
mse_lasso_train = mean_squared_error(y_train, y_train_pred)
mse_lasso_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error of Train:", mse_lasso_train)
print("Mean Squared Error of Test:", mse_lasso_test)

#Neural Network

tf.random.set_seed(329)
np.random.seed(329)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

nn_model.compile(loss = tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(lr=0.1),
                metrics=['mae','mse'])

history = nn_model.fit(X_train,y_train,epochs=200,verbose=2)

nn_model.summary()

#Evaluating the model

pd.DataFrame(history.history).plot()
plt.title('Loss Graph')
plt.ylabel('loss')
plt.xlabel('epochs');

y_pred = nn_model.predict(X_test)
y_train_pred = nn_model.predict(X_train)
r2_nn_train = r2_score(y_train, y_train_pred)
r2_nn_test = r2_score(y_test, y_pred)
print("R2 Train Score:", r2_nn_train)
print("R2 Test Score:", r2_nn_test)
mse_nn_train = mean_squared_error(y_train, y_train_pred)
mse_nn_test = mean_squared_error(y_test, y_pred )
print("Mean Squared Error of Train:", mse_nn_train)
print("Mean Squared Error of Test:", mse_nn_test)

from catboost import CatBoostRegressor
params = {'random_strength': 1,
 'n_estimators': 100,
 'max_depth': 7,
 'loss_function': 'RMSE',
 'learning_rate': 0.1,
 'colsample_bylevel': 0.8,
 'bootstrap_type': 'MVS',
 'bagging_temperature': 1.0}

model = CatBoostRegressor(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black', linewidth=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Catboost Regressor Predictions vs. True Values")
plt.show()

print("MSE: ", mean_squared_error(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='black', linewidth=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Catboost Regressor Predictions vs. True Values")
plt.show()

model_acc_test = r2_score(y_test , y_pred)

#Voting Regressor

pip install sklego

from sklego.linear_model import LADRegression
models = pd.DataFrame()
models["xgb"] = xgb.predict(X_test)
models["lrg"]= Lrg.predict(X_test)
models["rge"]=rge.predict(X_test)
models["roc"]=lso.predict(X_test)
models["rfc"]=rf.predict(X_test)
models["dt"]=dt.predict(X_test)
models["model"]=model.predict(X_test)
models["svr"]=svr.predict(X_test)
models["lasso"]=lasso.predict( X_test)
models["nn_model"]=nn_model(X_test)
weights = LADRegression().fit(models, y_test).coef_
pd.DataFrame(weights, index = models.columns, columns = ["weights"])

weights = LADRegression().fit(models, y_test).coef_

from sklearn.ensemble import VotingRegressor
voting = VotingRegressor(estimators=[('xgb',xgb),
                                      ('lrg',Lrg),
                                      ('rge',rge),
                                      ('roc',lso),
                                      ('rfc',rf),
                                      ('dt',dt),
                                      ('model',model),
                                      ('svr',svr),
                                      ('lasso',lasso),
                                      ('nn_model',nn_model)],weights=weights)
voting=xgb.fit(X_train,y_train)
voting_pred = voting.predict(X_test)

print('Error: ', mean_squared_error(y_test,y_pred, squared=False))

voting_pred = voting.predict(X_test)
voting_train_pred=voting.predict(X_train)
r2_voting_train = r2_score(y_train,voting_train_pred)
r2_votting_test = r2_score(y_test, voting_pred)
print("R2 Train Score:", r2_voting_train)
print("R2 Test Score:",r2_votting_test)
mse_voting_train = mean_squared_error(y_train, voting_train_pred)
mse_voting_test = mean_squared_error(y_test, voting_pred )
print("Mean Squared Error of Train:", mse_voting_train )
print("Mean Squared Error of Test:", mse_voting_test)

#Comparing modules

from sklearn.metrics import accuracy_score
models = pd.DataFrame({

    'Models': ['LinearRegression' , 'PolynomialFeatures','Ridge' , 'Lasso','RandomForestRegressor','DecisionTreeRegressor','XGBRegressor','CatBoostRegressor','SVR','RobustScaler','LADRegression'],
    'Scores': [reg_acc_test, r2_poly_test ,rge_acc_test,r2_lasso_test,rf_acc_test,dt_acc_test,xgb_acc_test,model_acc_test,r2_svr_lr_test ,lso_acc_test,r2_votting_test]

})

models.sort_values(by='Scores', ascending= False)

models.sort_values(by='Scores', ascending=False).style.background_gradient(
        cmap='Blues')

#Final prediction

df_train

df_test

df_test["Electrical"] = df_test["Electrical"].fillna("SBrkr")

df_test.drop(["GarageArea","BsmtFullBath","FullBath","TotRmsAbvGrd","2ndFlrSF","GarageYrBlt",'1stFlrSF'],axis=1,inplace=True)

df_test.drop(["PoolQC","Fence","Alley","MiscFeature"],axis=1,inplace=True)

df_test["MasVnrArea"] =df_test["MasVnrArea"].fillna(0)

df_test["LotFrontage"] = df_test["LotFrontage"].fillna(0)

df_test["MasVnrArea"] = df_test["MasVnrArea"].astype(int)

df_test["LotFrontage"] = df_test["LotFrontage"].astype(int)

cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
       
df_test[cols] = df_test[cols].apply(LabelEncoder().fit_transform)

(df_test.isnull().sum() / len(df) * 100).sort_values(ascending=False)

def predict_missing_values(test, model, target_column, exclude_columns=[]):
    # Prepare encoded DataFrame
    df_encoded = test.drop(exclude_columns + [target_column], axis=1, errors='ignore')

    # Label encode categorical columns
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype != 'object':
            df_encoded[col] = scaler.fit_transform(df_encoded[[col]])
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])

    # Predict missing values
    features_to_predict = df_encoded.drop([target_column], axis=1, errors='ignore')
    y_pred = model.predict(features_to_predict)

    # Replace missing values in the original DataFrame
    test[target_column] = y_pred

# Encoding and Scaling the data in df_test dataset
for col in df_test.columns:
        if df_test[col].dtype == 'object':
            df_test[col] = encoder.fit_transform(df_test[col])
        if df_test[col].dtype != 'object':
            df_test[col] = scaler.fit_transform(df_test[[col]])

df_test.head()

log_predictions = model.predict(df_test)

# Reverse the log transformation
predictions_1 = np.exp(log_predictions)

predections = model.predict(df_test)

predections

submisson = df_test[["Id"]]

df_train

test

test=test.drop(['1stFlrSF','2ndFlrSF','BsmtFullBath','FullBath','GarageYrBlt','GarageArea','GarageArea'],axis=1)

test=test.drop(['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'],axis=1)

test=test.drop(['ExterCond'],axis=1)

test=test.drop(['TotRmsAbvGrd','OverallCond','WoodDeckSF','OverallQual'],axis=1)

test

submission["SalePrice"]=np.exp(voting.predict(test))
submission.to_csv('submission.csv',index=False)
submission

submisson["SalePrice"] = predections

submisson

submisson.to_csv("submisson.csv" , index = False)

submission['SalePrice'] = predictions_1
submission.to_csv('submission.csv', index = False)

import pickle
with open("model-2" , "wb") as f:
    pickle.dump(lso , f)
