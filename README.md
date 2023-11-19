## 🌾🌵🌴 Logistic Regression 🌴🌵🌾
Using logistic regression to predict whether someone will buy after seeing an advertisement for a product.

'''
import pandas as pd
'''
 
## 🌾🌵🌴 Reading Dataset and Converting into Dataframe
converting the dataset into a Pandas dataframe. Don't forget to import the basic library and adjust the data path/location.
```
df = pd.read_csv('Social_Network_Ads.csv')
```

## 🌾🌵🌴 Using Head Function 🌴🌵🌾
In the next cell, use the head() function on the dataframe to see the first 5 rows of the dataset.
```
df.head()
```

The results of the df.head() function are as below.
![image](https://github.com/diantyapitaloka/Logistic-Regression/assets/147487436/c4443d0f-4f3b-4354-b470-419ee2bc65a9)

## 🌾🌵🌴 Find Duplicate Data 🌴🌵🌾
We also need to see whether there are empty values for each attribute by using the info() function. It can be seen that the values in all columns are complete.
```
df.info()
```

Meanwhile, the results of the df.info() function are displayed as follows.
![image](https://github.com/diantyapitaloka/Logistic-Regression/assets/147487436/1b599398-86ce-4017-a55d-6fba33b64e07)

## 🌾🌵🌴 Remove Useless Function 🌴🌵🌾
In the dataset there is a 'User ID' column. This column is an attribute that is not important for the model to learn so it needs to be removed. To remove a column from a dataframe, use the drop function. Don't forget to call the get_dummies() function to carry out the One-Hot Encoding process because the labels in our dataset are categorical data.

Drop columns that are not needed.
```
data = df.drop(columns=['User ID'])
```

Run the one-hot encoding process with 
```
pd.get_dummies()
data = pd.get_dummies(data)
data
```

When the above code is executed the results are as below.
![image](https://github.com/diantyapitaloka/Logistic-Regression/assets/147487436/9218057f-7ba6-4062-850f-49d513e15e39)

## 🌾🌵🌴 Showing Attributes and Labels 🌴🌵🌾
Then we separate attributes and labels.
```
predictions = ['Age' , 'EstimatedSalary' , 'Gender_Female' , 'Gender_Male']
X = data[predictions]
y = data['Purchased']
```

## 🌾🌵🌴 Standardize The Data 🌴🌵🌾
Before we divide the data into train and test sets, we need to standardize the data.
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
scaled_data.head()
```

## 🌾🌵🌴 Split The Data 🌴🌵🌾
Now, we will split the data into train and test sets with the train_test_split function provided by SKLearn.
```
from sklearn.model_selection import train_test_split
```
 
Split the data into train and test for each attribute and label
```
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)
```

After the model is trained, we can test the model's accuracy on the test set by calling the score() function on the model object.
```
model.score(X_test, y_test)
```

## 🌾🌵🌴 Output 🌴🌵🌾
So the results are as follows.

![image](https://github.com/diantyapitaloka/Logistic-Regression/assets/147487436/9ba02d07-d37d-445c-a5f6-3064e609c5a7)


