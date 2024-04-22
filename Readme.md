# Insights into Breast Cancer: Understanding through Exploratory Data Analysis
**Introduction:**
Breast cancer remains a pressing concern, impacting the lives of millions of women worldwide. Timely detection and accurate prognosis play pivotal roles in treatment outcomes and patient well-being.

In this notebook, we embark on an in-depth analysis of breast cancer data, focusing on clinical and demographic features such as age, tumor stage, lymph node involvement, tumor grade, and survival status. Our goal is to glean insights that can inform clinical decisions and treatment strategies.

By dissecting the intricate details of the dataset, we aim to shed light on patterns and correlations that may influence breast cancer detection and prognosis. Through this analysis, we strive to contribute to the collective understanding of breast cancer and its management, ultimately striving towards improved patient outcomes.

### Loading Dataset
We begin by loading the necessary libraries for data manipulation and visualization: NumPy, pandas, matplotlib, and seaborn. These libraries will help us load the dataset, explore its contents, and visualize key insights.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
data = pd.read_csv('Breast_Cancer.csv')
```


```python
data.sample(10)
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
      <th>Age</th>
      <th>Race</th>
      <th>Marital Status</th>
      <th>T Stage</th>
      <th>N Stage</th>
      <th>6th Stage</th>
      <th>differentiate</th>
      <th>Grade</th>
      <th>A Stage</th>
      <th>Tumor Size</th>
      <th>Estrogen Status</th>
      <th>Progesterone Status</th>
      <th>Regional Node Examined</th>
      <th>Reginol Node Positive</th>
      <th>Survival Months</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2747</th>
      <td>39</td>
      <td>White</td>
      <td>Divorced</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>35</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>6</td>
      <td>2</td>
      <td>44</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>879</th>
      <td>40</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>20</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>16</td>
      <td>1</td>
      <td>85</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>1403</th>
      <td>63</td>
      <td>White</td>
      <td>Single</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>20</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>13</td>
      <td>3</td>
      <td>95</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2450</th>
      <td>54</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>20</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>13</td>
      <td>2</td>
      <td>82</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2546</th>
      <td>44</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N3</td>
      <td>IIIC</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>40</td>
      <td>Negative</td>
      <td>Negative</td>
      <td>26</td>
      <td>21</td>
      <td>98</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2886</th>
      <td>47</td>
      <td>Black</td>
      <td>Single</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Well differentiated</td>
      <td>1</td>
      <td>Regional</td>
      <td>17</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>9</td>
      <td>1</td>
      <td>53</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>1662</th>
      <td>48</td>
      <td>White</td>
      <td>Single</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>24</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>20</td>
      <td>2</td>
      <td>106</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2504</th>
      <td>51</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>16</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>11</td>
      <td>5</td>
      <td>48</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>296</th>
      <td>45</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>20</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>1</td>
      <td>1</td>
      <td>81</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>52</td>
      <td>White</td>
      <td>Widowed</td>
      <td>T1</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>14</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>18</td>
      <td>5</td>
      <td>96</td>
      <td>Dead</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info
```




    <bound method DataFrame.info of       Age   Race Marital Status T Stage  N Stage 6th Stage  \
    0      68  White        Married       T1      N1       IIA   
    1      50  White        Married       T2      N2      IIIA   
    2      58  White       Divorced       T3      N3      IIIC   
    3      58  White        Married       T1      N1       IIA   
    4      47  White        Married       T2      N1       IIB   
    ...   ...    ...            ...      ...     ...       ...   
    4019   62  Other        Married       T1      N1       IIA   
    4020   56  White       Divorced       T2      N2      IIIA   
    4021   68  White        Married       T2      N1       IIB   
    4022   58  Black       Divorced       T2      N1       IIB   
    4023   46  White        Married       T2      N1       IIB   
    
                      differentiate Grade   A Stage  Tumor Size Estrogen Status  \
    0         Poorly differentiated     3  Regional           4        Positive   
    1     Moderately differentiated     2  Regional          35        Positive   
    2     Moderately differentiated     2  Regional          63        Positive   
    3         Poorly differentiated     3  Regional          18        Positive   
    4         Poorly differentiated     3  Regional          41        Positive   
    ...                         ...   ...       ...         ...             ...   
    4019  Moderately differentiated     2  Regional           9        Positive   
    4020  Moderately differentiated     2  Regional          46        Positive   
    4021  Moderately differentiated     2  Regional          22        Positive   
    4022  Moderately differentiated     2  Regional          44        Positive   
    4023  Moderately differentiated     2  Regional          30        Positive   
    
         Progesterone Status  Regional Node Examined  Reginol Node Positive  \
    0               Positive                      24                      1   
    1               Positive                      14                      5   
    2               Positive                      14                      7   
    3               Positive                       2                      1   
    4               Positive                       3                      1   
    ...                  ...                     ...                    ...   
    4019            Positive                       1                      1   
    4020            Positive                      14                      8   
    4021            Negative                      11                      3   
    4022            Positive                      11                      1   
    4023            Positive                       7                      2   
    
          Survival Months Status  
    0                  60  Alive  
    1                  62  Alive  
    2                  75  Alive  
    3                  84  Alive  
    4                  50  Alive  
    ...               ...    ...  
    4019               49  Alive  
    4020               69  Alive  
    4021               69  Alive  
    4022               72  Alive  
    4023              100  Alive  
    
    [4024 rows x 16 columns]>




```python
data.describe()
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
      <th>Age</th>
      <th>Tumor Size</th>
      <th>Regional Node Examined</th>
      <th>Reginol Node Positive</th>
      <th>Survival Months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.972167</td>
      <td>30.473658</td>
      <td>14.357107</td>
      <td>4.158052</td>
      <td>71.297962</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.963134</td>
      <td>21.119696</td>
      <td>8.099675</td>
      <td>5.109331</td>
      <td>22.921430</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.000000</td>
      <td>16.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>56.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>25.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>38.000000</td>
      <td>19.000000</td>
      <td>5.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>69.000000</td>
      <td>140.000000</td>
      <td>61.000000</td>
      <td>46.000000</td>
      <td>107.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Cleaning the Data

Cleaning the dataset is a crucial step to ensure its accuracy and reliability for analysis. In this section, we perform various cleaning operations to remove duplicates and verify the integrity of the dataset.



```python
data.isnull().sum()
```




    Age                       0
    Race                      0
    Marital Status            0
    T Stage                   0
    N Stage                   0
    6th Stage                 0
    differentiate             0
    Grade                     0
    A Stage                   0
    Tumor Size                0
    Estrogen Status           0
    Progesterone Status       0
    Regional Node Examined    0
    Reginol Node Positive     0
    Survival Months           0
    Status                    0
    dtype: int64




```python
data.rename(columns={'Reginol Node Positive' : 'Regional Node Positive'}, inplace=True)
```


```python
data
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
      <th>Age</th>
      <th>Race</th>
      <th>Marital Status</th>
      <th>T Stage</th>
      <th>N Stage</th>
      <th>6th Stage</th>
      <th>differentiate</th>
      <th>Grade</th>
      <th>A Stage</th>
      <th>Tumor Size</th>
      <th>Estrogen Status</th>
      <th>Progesterone Status</th>
      <th>Regional Node Examined</th>
      <th>Regional Node Positive</th>
      <th>Survival Months</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>4</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>24</td>
      <td>1</td>
      <td>60</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>35</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>5</td>
      <td>62</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>White</td>
      <td>Divorced</td>
      <td>T3</td>
      <td>N3</td>
      <td>IIIC</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>63</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>7</td>
      <td>75</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>18</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>2</td>
      <td>1</td>
      <td>84</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>41</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>3</td>
      <td>1</td>
      <td>50</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4019</th>
      <td>62</td>
      <td>Other</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>9</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>1</td>
      <td>1</td>
      <td>49</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4020</th>
      <td>56</td>
      <td>White</td>
      <td>Divorced</td>
      <td>T2</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>46</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>8</td>
      <td>69</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4021</th>
      <td>68</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>22</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>11</td>
      <td>3</td>
      <td>69</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4022</th>
      <td>58</td>
      <td>Black</td>
      <td>Divorced</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>44</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>11</td>
      <td>1</td>
      <td>72</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4023</th>
      <td>46</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>30</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>7</td>
      <td>2</td>
      <td>100</td>
      <td>Alive</td>
    </tr>
  </tbody>
</table>
<p>4024 rows × 16 columns</p>
</div>



#### Delete Duplicates

Duplicate records can skew analysis results and lead to erroneous conclusions. We identify and remove duplicate rows from the dataset.



```python
data[data.duplicated()]
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
      <th>Age</th>
      <th>Race</th>
      <th>Marital Status</th>
      <th>T Stage</th>
      <th>N Stage</th>
      <th>6th Stage</th>
      <th>differentiate</th>
      <th>Grade</th>
      <th>A Stage</th>
      <th>Tumor Size</th>
      <th>Estrogen Status</th>
      <th>Progesterone Status</th>
      <th>Regional Node Examined</th>
      <th>Regional Node Positive</th>
      <th>Survival Months</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>436</th>
      <td>63</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>17</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>9</td>
      <td>1</td>
      <td>56</td>
      <td>Alive</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop_duplicates()
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
      <th>Age</th>
      <th>Race</th>
      <th>Marital Status</th>
      <th>T Stage</th>
      <th>N Stage</th>
      <th>6th Stage</th>
      <th>differentiate</th>
      <th>Grade</th>
      <th>A Stage</th>
      <th>Tumor Size</th>
      <th>Estrogen Status</th>
      <th>Progesterone Status</th>
      <th>Regional Node Examined</th>
      <th>Regional Node Positive</th>
      <th>Survival Months</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>4</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>24</td>
      <td>1</td>
      <td>60</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>35</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>5</td>
      <td>62</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>White</td>
      <td>Divorced</td>
      <td>T3</td>
      <td>N3</td>
      <td>IIIC</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>63</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>7</td>
      <td>75</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58</td>
      <td>White</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>18</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>2</td>
      <td>1</td>
      <td>84</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Poorly differentiated</td>
      <td>3</td>
      <td>Regional</td>
      <td>41</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>3</td>
      <td>1</td>
      <td>50</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4019</th>
      <td>62</td>
      <td>Other</td>
      <td>Married</td>
      <td>T1</td>
      <td>N1</td>
      <td>IIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>9</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>1</td>
      <td>1</td>
      <td>49</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4020</th>
      <td>56</td>
      <td>White</td>
      <td>Divorced</td>
      <td>T2</td>
      <td>N2</td>
      <td>IIIA</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>46</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>14</td>
      <td>8</td>
      <td>69</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4021</th>
      <td>68</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>22</td>
      <td>Positive</td>
      <td>Negative</td>
      <td>11</td>
      <td>3</td>
      <td>69</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4022</th>
      <td>58</td>
      <td>Black</td>
      <td>Divorced</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>44</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>11</td>
      <td>1</td>
      <td>72</td>
      <td>Alive</td>
    </tr>
    <tr>
      <th>4023</th>
      <td>46</td>
      <td>White</td>
      <td>Married</td>
      <td>T2</td>
      <td>N1</td>
      <td>IIB</td>
      <td>Moderately differentiated</td>
      <td>2</td>
      <td>Regional</td>
      <td>30</td>
      <td>Positive</td>
      <td>Positive</td>
      <td>7</td>
      <td>2</td>
      <td>100</td>
      <td>Alive</td>
    </tr>
  </tbody>
</table>
<p>4023 rows × 16 columns</p>
</div>



#### Check Dataset

After removing duplicates, it's essential to verify the integrity of the dataset to ensure that all cleaning operations have been successfully applied.



```python
data.describe()
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
      <th>Age</th>
      <th>Tumor Size</th>
      <th>Regional Node Examined</th>
      <th>Regional Node Positive</th>
      <th>Survival Months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
      <td>4024.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.972167</td>
      <td>30.473658</td>
      <td>14.357107</td>
      <td>4.158052</td>
      <td>71.297962</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.963134</td>
      <td>21.119696</td>
      <td>8.099675</td>
      <td>5.109331</td>
      <td>22.921430</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.000000</td>
      <td>16.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>56.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>25.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>38.000000</td>
      <td>19.000000</td>
      <td>5.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>69.000000</td>
      <td>140.000000</td>
      <td>61.000000</td>
      <td>46.000000</td>
      <td>107.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Descriptive Analysis

In this section, we conduct a descriptive analysis of the dataset to gain insights into key variables and characteristics.



```python
print(data.describe())
```

                   Age   Tumor Size  Regional Node Examined  \
    count  4024.000000  4024.000000             4024.000000   
    mean     53.972167    30.473658               14.357107   
    std       8.963134    21.119696                8.099675   
    min      30.000000     1.000000                1.000000   
    25%      47.000000    16.000000                9.000000   
    50%      54.000000    25.000000               14.000000   
    75%      61.000000    38.000000               19.000000   
    max      69.000000   140.000000               61.000000   
    
           Regional Node Positive  Survival Months  
    count             4024.000000      4024.000000  
    mean                 4.158052        71.297962  
    std                  5.109331        22.921430  
    min                  1.000000         1.000000  
    25%                  1.000000        56.000000  
    50%                  2.000000        73.000000  
    75%                  5.000000        90.000000  
    max                 46.000000       107.000000  
    

#### Calculate mean, min, and max values for specific variables


```python
meanAge = data['Age'].mean()
minAge = data['Age'].min()
maxAge = data['Age'].max()

print(f"Mean Age: {meanAge:.2f} years")
print(f"Minimum Age: {minAge} years")
print(f"Maximum Age: {maxAge} years")
```

    Mean Age: 53.97 years
    Minimum Age: 30 years
    Maximum Age: 69 years
    


```python
meanTumorSize = data['Tumor Size'].mean()
minTumorSize = data['Tumor Size'].min()
maxTumorSize = data['Tumor Size'].max()

print(f"Mean Tumor Size: {meanTumorSize:.2f} mm")
print(f"Minimum Tumor Size: {minTumorSize} mm")
print(f"Maximum Tumor Size: {maxTumorSize} mm")
```

    Mean Tumor Size: 30.47 mm
    Minimum Tumor Size: 1 mm
    Maximum Tumor Size: 140 mm
    


```python
meanRegionalNodeExamined = data['Regional Node Examined'].mean()
minRegionalNodeExamined = data['Regional Node Examined'].min()
maxRegionalNodeExamined = data['Regional Node Examined'].max()

print(f"Mean Regional Nodes Examined: {meanRegionalNodeExamined:.2f}")
print(f"Minimum Regional Nodes Examined: {minRegionalNodeExamined}")
print(f"Maximum Regional Nodes Examined: {maxRegionalNodeExamined}")
```

    Mean Regional Nodes Examined: 14.36
    Minimum Regional Nodes Examined: 1
    Maximum Regional Nodes Examined: 61
    


```python
meanRegionalNodePositive = data['Regional Node Positive'].mean()
minRegionalNodePositive = data['Regional Node Positive'].min()
maxRegionalNodePositive = data['Regional Node Positive'].max()

print(f"Mean Regional Nodes Positive: {meanRegionalNodePositive:.2f}")
print(f"Minimum Regional Nodes Positive: {minRegionalNodePositive}")
print(f"Maximum Regional Nodes Positive: {maxRegionalNodePositive}")
```

    Mean Regional Nodes Positive: 4.16
    Minimum Regional Nodes Positive: 1
    Maximum Regional Nodes Positive: 46
    


```python
meanSurvivalMonths = data['Survival Months'].mean()
minSurvivalMonths = data['Survival Months'].min()
maxSurvivalMonths = data['Survival Months'].max()

print(f"Mean Survival Months: {meanSurvivalMonths:.2f}")
print(f"Minimum Survival Months: {minSurvivalMonths}")
print(f"Maximum Survival Months: {maxSurvivalMonths}")
```

    Mean Survival Months: 71.30
    Minimum Survival Months: 1
    Maximum Survival Months: 107
    

#### Calculate mortality rate


```python
mortalityRate = ((data['Status']=='Dead').sum() / data.shape[0]) * 1000
print(f'Mortality Rate is {round(mortalityRate,2)} per 1000 person')
```

    Mortality Rate is 153.08 per 1000 person
    

#### Frequency of race categories


```python
freqInBlack = (data['Race'] == 'Black').sum()
freqInWhite = (data['Race'] == 'White').sum()
freqInOther = (data['Race'] == 'Other').sum()

print(f"Frequency of Black Race: {freqInBlack}")
print(f"Frequency of White Race: {freqInWhite}")
print(f"Frequency of Other Race: {freqInOther}")
```

    Frequency of Black Race: 291
    Frequency of White Race: 3413
    Frequency of Other Race: 320
    

### Data Visualization

In this section, we visualize key aspects of the dataset to gain insights and understand patterns.



```python
sns.set_style('darkgrid')
```

#### Plot a histogram of patient ages


```python
plt.hist(data['Age'], bins=25)
plt.ylabel('Number of patients')
plt.xlabel('Age')
plt.title('Age vs Number of Patients Graph');
```


    
![png](output_31_0.png)
    


**Insight:** Most patients are around the age of 45 to 65.


```python
plt.hist(data['Marital Status'])
;
```




    ''




    
![png](output_33_1.png)
    


**Insight:** Most of the breast cancer occurred in married patients.


```python
plt.hist(data['Race']);
```


    
![png](output_35_0.png)
    


**Insight:** Majority of patients were white, with some black and some other races.


```python
race_counts = data['Race'].value_counts()
plt.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Race Distribution Among Patients')
plt.axis('equal')
plt.show()

```


    
![png](output_37_0.png)
    



```python
plt.pie(data['Status'].value_counts(), labels=data['Status'].value_counts().index, autopct='%1.1f%%')
plt.title('Dead vs Alive distribution')
plt.axis('equal')
plt.show()
```


    
![png](output_38_0.png)
    



```python
plt.figure(figsize=(10,10))
sns.scatterplot(x='Tumor Size', y='Survival Months',hue='Status', data=data)
plt.show()
```


    
![png](output_39_0.png)
    


**Insight:** Most common tumor size is around 10 to 30, and most deaths occur in this range. Larger tumors are rare.


```python
sns.scatterplot(x='Regional Node Examined', y='Regional Node Positive', hue='Status', data=data)
plt.show()
```


    
![png](output_41_0.png)
    


**Insight:** As more regional nodes are examined, more positive nodes are found. Higher number of deaths are observed for higher number of regional nodes examined.


```python
sns.countplot(x='differentiate', data=data)
plt.xticks(rotation=25)
plt.show()
```


    
![png](output_43_0.png)
    



```python
fig,axes = plt.subplots(1,4,figsize=(16,4))

unDifferentiated = data[data['differentiate'] == 'Undifferentiated']
unDifferentiatedStatus = unDifferentiated['Status'].value_counts()
axes[0].pie(unDifferentiatedStatus, labels=unDifferentiatedStatus.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Undifferentiated')

poorlyDifferentiated = data[data['differentiate']=='Poorly differentiated']
poorlyDifferentiatedStatus = poorlyDifferentiated['Status'].value_counts()
axes[1].pie(poorlyDifferentiatedStatus, labels=poorlyDifferentiatedStatus.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Poorly Differentiate')

moderatelyDifferentiated = data[data['differentiate'] == 'Moderately differentiated']
moderatelyDifferentiatedStatus = moderatelyDifferentiated['Status'].value_counts()
axes[2].pie(moderatelyDifferentiatedStatus, labels=moderatelyDifferentiatedStatus.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Moderately Differentiated')

wellDifferentiated = data[data['differentiate'] == 'Well differentiated']
wellDifferentiatedStatus = wellDifferentiated['Status'].value_counts()
axes[3].pie(wellDifferentiatedStatus, labels=wellDifferentiatedStatus.index, autopct='%1.1f%%', startangle=90)
axes[3].set_title('Well Differentiated')

plt.show()
```


    
![png](output_44_0.png)
    


**Insight:** Most samples are moderately differentiated, while poorly differentiated samples are half of that, and even fewer are differentiated. Undifferentiated samples are uncommon.
Comparing mortality with differentiation, we found that undifferentiated patients had the highest mortality rate of 47.4%, while well-differentiated had the lowest of 7.2%.



```python
sns.countplot(x='T Stage ',hue='Status', data=data)
```




    <Axes: xlabel='T Stage ', ylabel='count'>




    
![png](output_46_1.png)
    


**Insight:** Most number of patients were in T1 or T2 stage, with the least in T4. In comparison, the survival rate of T1 patients was the highest, and T4 the lowest.

### Conclusion

In this analysis, we explored a dataset containing information about breast cancer patients, including demographic characteristics, tumor attributes, treatment factors, and survival outcomes. Through descriptive analysis and data visualization, we gained valuable insights into various aspects of breast cancer diagnosis, treatment, and prognosis.

**Key Findings:**

1. **Demographic Insights:**
   - Most patients in the dataset were married, and the majority were of white race.
   - The age distribution of patients ranged from approximately 20 to 90 years, with a peak in the age range of 45 to 65 years.

2. **Clinical Characteristics:**
   - Tumor size varied widely, with the most common tumor sizes falling in the range of 10 to 30 units.
   - Examination of regional lymph nodes revealed a positive correlation between the number of nodes examined and the number of positive nodes found.
   - Differentiation status showed that most samples were moderately differentiated, followed by poorly differentiated and well-differentiated samples. Undifferentiated samples were less common.

3. **Survival Analysis:**
   - The survival months after diagnosis ranged from a few months to several years, with varying outcomes.
   - Mortality rate analysis revealed an overall mortality rate of 153 per 1000 persons.
   - Patients with undifferentiated tumors exhibited the highest mortality rate, while those with well-differentiated tumors had the lowest mortality rate.

4. **Treatment and Prognosis:**
   - The majority of patients were diagnosed at T1 or T2 stage, with T1 patients showing the highest survival rate and T4 patients showing the lowest.

**Implications:**

- The findings from this analysis provide valuable insights for healthcare professionals, researchers, and policymakers involved in breast cancer prevention, diagnosis, and treatment.
- Understanding the demographic and clinical characteristics of breast cancer patients can aid in the development of personalized treatment strategies and interventions.
- Further research is warranted to explore the underlying factors contributing to variations in survival outcomes and to identify novel biomarkers or therapeutic targets for improved patient management.

**Limitations:**

- The analysis is based on a single dataset and may not capture the full spectrum of breast cancer cases or account for regional or temporal variations.
- The dataset may contain inherent biases or limitations due to data collection methods or missing information.

**Future Directions:**

- Future studies could explore additional factors such as genetic mutations, lifestyle factors, or environmental exposures to further elucidate the etiology and progression of breast cancer.
- Longitudinal studies tracking patient outcomes over time could provide valuable insights into the long-term effects of different treatment modalities and interventions.

In conclusion, this analysis contributes to our understanding of breast cancer epidemiology, diagnosis, and treatment outcomes. By leveraging data-driven approaches, we can continue to advance our knowledge and improve patient care in the fight against breast cancer.

---

### Author

This analysis was conducted by Swastik Tripathi, a student of Computer Science and Engineering.

For any inquiries or further discussions, feel free to reach out via email at [swastiktripathi.space@gmail.com](mailto:swastiktripathi.space@gmail.com).



```python

```
