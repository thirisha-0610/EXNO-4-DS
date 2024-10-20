# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/ac49ac9d-8b6c-482d-baf6-6b52b1a87a32)
```
df.head()
```
![image](https://github.com/user-attachments/assets/aabc4abb-f526-4e58-a2b7-216bc0f72f30)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/8efd4adf-e3b3-4a79-8b59-0e00d5507856)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/a3f035ec-4188-4a7a-9bb1-16f29b54fb69)
```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
```
```
df.head(10)
```
![image](https://github.com/user-attachments/assets/9d177a13-f07a-4d39-85f4-8d69b4810851)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/8b83700c-9d04-458a-803b-273079d16656)
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/7c9fc9ab-faaa-4400-aba1-f687447cfeec)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/eac6c0b4-5925-4fc4-895d-a025ab7b2109)
```
import pandas as pd
import numpy as np
import seaborn as sns
```
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/db87ed08-8ba8-4776-a0f1-a06d689a67fb)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/c81ca1c9-70d4-47b7-834b-2bca0ff66095)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/88f8551d-b2d6-41eb-9391-bc157256f883)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/13c081d4-1338-48df-8353-fc5e579b0948)
```
sal=data['SalStat']
```
```
data2['SalStat']=data['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/7eba186e-0379-46fb-a2ce-b954fb9cf011)
```
sal2=data2['SalStat']
```
```
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/b4911fde-3b57-404b-ad0d-413bf015e857)
```
data2
```
![image](https://github.com/user-attachments/assets/418ec259-4f69-49fb-b89d-327736f50f5c)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/423492da-00cb-46f5-bc75-3fa0d8b84bfc)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/f926dc7a-74ec-48c3-a841-acfe762c047a)
```
features=list(set(columns_list))
print(features)
```
![image](https://github.com/user-attachments/assets/2824bfac-2d56-4864-8b23-c17e2212ed2a)
```
y=new_data['SalStat'].values
y
```
![image](https://github.com/user-attachments/assets/9284e4b8-d895-41cb-8a07-6c67b3af3250)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/5f8162e8-e853-4b55-8c72-5b6bc224ca9c)
ALGORITHM IMPLEMENTATION
```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
```
```
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/c694772e-2c62-4227-8cea-4f78ce39ed0d)
```
prediction=KNN_classifier.predict(test_x)
```
```
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/0a7b1542-0ccd-49b2-9f5f-15ea179f0930)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/0513ff7f-d6c9-4f0e-b6f6-bb54c556b706)
```
data.shape
```
![image](https://github.com/user-attachments/assets/645a9c80-dba1-4910-8d1d-a1621f97ae16)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
