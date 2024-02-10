```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.read_csv("https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (550068, 12)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 550068 entries, 0 to 550067
    Data columns (total 12 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   User_ID                     550068 non-null  int64  
     1   Product_ID                  550068 non-null  object 
     2   Gender                      550068 non-null  object 
     3   Age                         550068 non-null  object 
     4   Occupation                  550068 non-null  int64  
     5   City_Category               550068 non-null  object 
     6   Stay_In_Current_City_Years  550068 non-null  object 
     7   Marital_Status              550068 non-null  int64  
     8   Product_Category_1          550068 non-null  int64  
     9   Product_Category_2          376430 non-null  float64
     10  Product_Category_3          166821 non-null  float64
     11  Purchase                    550068 non-null  int64  
    dtypes: float64(2), int64(5), object(5)
    memory usage: 50.4+ MB
    


```python
data.isnull().sum()
```




    User_ID                            0
    Product_ID                         0
    Gender                             0
    Age                                0
    Occupation                         0
    City_Category                      0
    Stay_In_Current_City_Years         0
    Marital_Status                     0
    Product_Category_1                 0
    Product_Category_2            173638
    Product_Category_3            383247
    Purchase                           0
    dtype: int64




```python
data.isnull().sum()/data.shape[0]*100
```




    User_ID                        0.000000
    Product_ID                     0.000000
    Gender                         0.000000
    Age                            0.000000
    Occupation                     0.000000
    City_Category                  0.000000
    Stay_In_Current_City_Years     0.000000
    Marital_Status                 0.000000
    Product_Category_1             0.000000
    Product_Category_2            31.566643
    Product_Category_3            69.672659
    Purchase                       0.000000
    dtype: float64




```python
data.nunique()
```




    User_ID                        5891
    Product_ID                     3631
    Gender                            2
    Age                               7
    Occupation                       21
    City_Category                     3
    Stay_In_Current_City_Years        5
    Marital_Status                    2
    Product_Category_1               20
    Product_Category_2               17
    Product_Category_3               15
    Purchase                      18105
    dtype: int64




```python
sns.distplot(data["Purchase"],color='r')
plt.title("Purchase Distribution")
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


    
![png](output_8_1.png)
    



```python
sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_9_1.png)
    



```python
data["Purchase"].skew()
```




    0.6001400037087128




```python
data["Purchase"].kurtosis()
```




    -0.3383775655851702




```python
data["Purchase"].describe()
```




    count    550068.000000
    mean       9263.968713
    std        5023.065394
    min          12.000000
    25%        5823.000000
    50%        8047.000000
    75%       12054.000000
    max       23961.000000
    Name: Purchase, dtype: float64




```python
sns.countplot(data['Gender'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_13_1.png)
    



```python
data['Gender'].value_counts(normalize=True)*100
```




    M    75.310507
    F    24.689493
    Name: Gender, dtype: float64




```python
data['Gender'].value_counts(normalize=True)*100
```




    M    75.310507
    F    24.689493
    Name: Gender, dtype: float64




```python
data.groupby("Gender").mean()["Purchase"]
```




    Gender
    F    8734.565765
    M    9437.526040
    Name: Purchase, dtype: float64




```python
sns.countplot(data['Marital_Status'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_17_1.png)
    



```python
data.groupby("Marital_Status").mean()["Purchase"]
```




    Marital_Status
    0    9265.907619
    1    9261.174574
    Name: Purchase, dtype: float64




```python
data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
plt.title("Marital_Status and Purchase Analysis")
plt.show()
```


    
![png](output_19_0.png)
    



```python
plt.figure(figsize=(18,5))
sns.countplot(data['Occupation'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_20_1.png)
    



```python
occup = pd.DataFrame(data.groupby("Occupation").mean()["Purchase"])
occup
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase</th>
    </tr>
    <tr>
      <th>Occupation</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9124.428588</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8953.193270</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8952.481683</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9178.593088</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9213.980251</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9333.149298</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9256.535691</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9425.728223</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9532.592497</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8637.743761</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8959.355375</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9213.845848</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9796.640239</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9306.351061</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9500.702772</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9778.891163</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9394.464349</td>
    </tr>
    <tr>
      <th>17</th>
      <td>9821.478236</td>
    </tr>
    <tr>
      <th>18</th>
      <td>9169.655844</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8710.627231</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8836.494905</td>
    </tr>
  </tbody>
</table>
</div>




```python
occup.plot(kind='bar',figsize=(15,5))
plt.title("Occupation and Purchase Analysis")
plt.show()
```


    
![png](output_22_0.png)
    



```python
sns.countplot(data['City_Category'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_23_1.png)
    



```python
data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
plt.title("City Category and Purchase Analysis")
plt.show()
```


    
![png](output_24_0.png)
    



```python
sns.countplot(data['Stay_In_Current_City_Years'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_25_1.png)
    



```python
data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
plt.title("Stay_In_Current_City_Years and Purchase Analysis")
plt.show()
```


    
![png](output_26_0.png)
    



```python
sns.countplot(data['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_27_1.png)
    



```python
data.groupby("Age").mean()["Purchase"].plot(kind='bar')
```




    <AxesSubplot:xlabel='Age'>




    
![png](output_28_1.png)
    



```python
data.groupby("Age").sum()['Purchase'].plot(kind="bar")
plt.title("Age and Purchase Analysis")
plt.show()
```


    
![png](output_29_0.png)
    



```python
plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_1'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_30_1.png)
    



```python
data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()
```


    
![png](output_31_0.png)
    



```python
data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Analysis")
plt.show()
```


    
![png](output_32_0.png)
    



```python
plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_2'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_33_1.png)
    



```python
plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_3'])
plt.show()
```

    C:\Users\lbhav\anaconda2\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_34_1.png)
    



```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Occupation</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User_ID</th>
      <td>1.000000</td>
      <td>-0.023971</td>
      <td>0.020443</td>
      <td>0.003825</td>
      <td>0.001529</td>
      <td>0.003419</td>
      <td>0.004716</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>-0.023971</td>
      <td>1.000000</td>
      <td>0.024280</td>
      <td>-0.007618</td>
      <td>-0.000384</td>
      <td>0.013263</td>
      <td>0.020833</td>
    </tr>
    <tr>
      <th>Marital_Status</th>
      <td>0.020443</td>
      <td>0.024280</td>
      <td>1.000000</td>
      <td>0.019888</td>
      <td>0.015138</td>
      <td>0.019473</td>
      <td>-0.000463</td>
    </tr>
    <tr>
      <th>Product_Category_1</th>
      <td>0.003825</td>
      <td>-0.007618</td>
      <td>0.019888</td>
      <td>1.000000</td>
      <td>0.540583</td>
      <td>0.229678</td>
      <td>-0.343703</td>
    </tr>
    <tr>
      <th>Product_Category_2</th>
      <td>0.001529</td>
      <td>-0.000384</td>
      <td>0.015138</td>
      <td>0.540583</td>
      <td>1.000000</td>
      <td>0.543649</td>
      <td>-0.209918</td>
    </tr>
    <tr>
      <th>Product_Category_3</th>
      <td>0.003419</td>
      <td>0.013263</td>
      <td>0.019473</td>
      <td>0.229678</td>
      <td>0.543649</td>
      <td>1.000000</td>
      <td>-0.022006</td>
    </tr>
    <tr>
      <th>Purchase</th>
      <td>0.004716</td>
      <td>0.020833</td>
      <td>-0.000463</td>
      <td>-0.343703</td>
      <td>-0.209918</td>
      <td>-0.022006</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(data.corr(),annot=True)
plt.show()
```


    
![png](output_36_0.png)
    



```python
data.columns
```




    Index(['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
           'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
           'Product_Category_2', 'Product_Category_3', 'Purchase'],
          dtype='object')




```python
df = data.copy()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Stay_In_Current_City_Years</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>F</td>
      <td>0-17</td>
      <td>10</td>
      <td>A</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>M</td>
      <td>55+</td>
      <td>16</td>
      <td>C</td>
      <td>4+</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace="4+",value="4")
```


```python

df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])
```


```python
from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
```


```python
df['Gender'] = lr.fit_transform(df['Gender'])
```


```python
df['Age'] = lr.fit_transform(df['Age'])
```


```python
df['City_Category'] = lr.fit_transform(df['City_Category'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Product_ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>City_Category</th>
      <th>Marital_Status</th>
      <th>Product_Category_1</th>
      <th>Product_Category_2</th>
      <th>Product_Category_3</th>
      <th>Purchase</th>
      <th>Stay_In_Current_City_Years_0</th>
      <th>Stay_In_Current_City_Years_1</th>
      <th>Stay_In_Current_City_Years_2</th>
      <th>Stay_In_Current_City_Years_3</th>
      <th>Stay_In_Current_City_Years_4+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000001</td>
      <td>P00069042</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8370</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000001</td>
      <td>P00248942</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>15200</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000001</td>
      <td>P00087842</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1422</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000001</td>
      <td>P00085442</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>1057</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000002</td>
      <td>P00285442</td>
      <td>1</td>
      <td>6</td>
      <td>16</td>
      <td>2</td>
      <td>0</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7969</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')
```


```python
df.isnull().sum()
```




    User_ID                          0
    Product_ID                       0
    Gender                           0
    Age                              0
    Occupation                       0
    City_Category                    0
    Marital_Status                   0
    Product_Category_1               0
    Product_Category_2               0
    Product_Category_3               0
    Purchase                         0
    Stay_In_Current_City_Years_0     0
    Stay_In_Current_City_Years_1     0
    Stay_In_Current_City_Years_2     0
    Stay_In_Current_City_Years_3     0
    Stay_In_Current_City_Years_4+    0
    dtype: int64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 550068 entries, 0 to 550067
    Data columns (total 16 columns):
     #   Column                         Non-Null Count   Dtype 
    ---  ------                         --------------   ----- 
     0   User_ID                        550068 non-null  int64 
     1   Product_ID                     550068 non-null  object
     2   Gender                         550068 non-null  int32 
     3   Age                            550068 non-null  int32 
     4   Occupation                     550068 non-null  int64 
     5   City_Category                  550068 non-null  int32 
     6   Marital_Status                 550068 non-null  int64 
     7   Product_Category_1             550068 non-null  int64 
     8   Product_Category_2             550068 non-null  int64 
     9   Product_Category_3             550068 non-null  int64 
     10  Purchase                       550068 non-null  int64 
     11  Stay_In_Current_City_Years_0   550068 non-null  uint8 
     12  Stay_In_Current_City_Years_1   550068 non-null  uint8 
     13  Stay_In_Current_City_Years_2   550068 non-null  uint8 
     14  Stay_In_Current_City_Years_3   550068 non-null  uint8 
     15  Stay_In_Current_City_Years_4+  550068 non-null  uint8 
    dtypes: int32(3), int64(7), object(1), uint8(5)
    memory usage: 42.5+ MB
    


```python
df = df.drop(["User_ID","Product_ID"],axis=1)
```


```python
X = df.drop("Purchase",axis=1)
```


```python
y=df['Purchase']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
```


```python
from sklearn.linear_model import LinearRegression
```


```python
lr = LinearRegression()
lr.fit(X_train,y_train)
```




    LinearRegression()




```python
lr.intercept_
```




    9536.400764131557




```python
lr.coef_
```




    array([ 465.82318446,  112.36643445,    5.05508596,  314.06766138,
            -58.23217776, -348.4514785 ,   12.98415047,  143.49190467,
            -20.83796687,    5.4676518 ,   17.68367185,   -3.96751734,
              1.65416056])




```python
y_pred = lr.predict(X_test)
```


```python
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
```


```python
mean_absolute_error(y_test, y_pred)
```




    3532.0692261658432




```python
mean_squared_error(y_test, y_pred)
```




    21397853.26940752




```python
mean_squared_error(y_test, y_pred)
```




    21397853.26940752




```python
r2_score(y_test, y_pred)
```




    0.15192944521481666




```python
from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, y_pred)))
```

    RMSE of Linear Regression Model is  4625.781368526567
    


```python
from sklearn.tree import DecisionTreeRegressor


regressor = DecisionTreeRegressor(random_state = 0)  
```


```python
regressor.fit(X_train, y_train)
```




    DecisionTreeRegressor(random_state=0)




```python
dt_y_pred = regressor.predict(X_test)
```


```python
mean_absolute_error(y_test, dt_y_pred)
```




    2372.0357559134654




```python
mean_squared_error(y_test, dt_y_pred)
```




    11300579.466797074




```python
r2_score(y_test, dt_y_pred)
```




    0.5521191505924365




```python
from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, dt_y_pred)))
```

    RMSE of Linear Regression Model is  3361.633452177241
    


```python
from sklearn.ensemble import RandomForestRegressor

# create a regressor object 
RFregressor = RandomForestRegressor(random_state = 0)  
```


```python
RFregressor.fit(X_train, y_train)
```




    RandomForestRegressor(random_state=0)




```python
rf_y_pred = RFregressor.predict(X_test)
```


```python
mean_absolute_error(y_test, rf_y_pred)
```




    2222.049109204734




```python
mean_squared_error(y_test, rf_y_pred)
```




    9310769.87311957




```python
r2_score(y_test, rf_y_pred)
```




    0.6309821516972987




```python
from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, rf_y_pred)))
```

    RMSE of Linear Regression Model is  3051.35541573242
    


```python

```
