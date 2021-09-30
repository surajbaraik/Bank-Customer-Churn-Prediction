# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:13:38 2021

@author: suraj baraik
"""
import pandas as pd
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the Dataset
df = pd.read_csv('C:\\Users\\suraj baraik\\Desktop\\Sutherland\\Customer Churn\\Dataset\\Churn_Modelling.csv')
df.head()
df.shape
df.dtypes

# Check columns list and missing values
df.isnull().sum()

df.nunique()
# Printing Unique Values of the categorical variables
print(df['Geography'].unique())
print(df['Gender'].unique())
print(df['NumOfProducts'].unique())
print(df['HasCrCard'].unique())
print(df['IsActiveMember'].unique())

df.describe()
df.head()
##### Churn column is the dependent variable(Y) 
### customer ID, Surname and Row number are not relevant to Y, so we can drop.
### rest of the other are Potential Predictors as independent varibles (X)

df = df.drop(["RowNumber", "CustomerId","Surname"], axis = 1)
df.head()
df.dtypes

#### EDA 
labels = 'Exited', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])

'''
We note the following:

1 - Majority of the data is from persons from France. However, the proportion of churned customers is with 
    inversely related to the population of customers alluding to the bank possibly having a 
    problem (maybe not enough customer service resources allocated) in the areas where it has fewer clients.
2-  The proportion of female customers churning is also greater than that of male customers
3-  Interestingly, majority of the customers that churned are those with credit cards. Given that majority of the customers 
    have credit cards could prove this to be just a coincidence.
4-  Unsurprisingly the inactive members have a greater churn. Worryingly is that the overall proportion of inactive 
    mebers is quite high suggesting that the bank may need a program implemented to turn this 
    group to active customers as this will definately have a positive impact on the customer churn.
    
'''

# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(15, 10))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])

'''
We note the following:

1. There is no significant difference in the credit score distribution between retained and churned customers.
2. The older customers are churning at more than the younger ones alluding to a difference in service 
    preference in the age categories. The bank may need to review their target market or review the 
    strategy for retention between the different age groups
3. With regard to the tenure, the clients on either extreme end (spent little time with the bank 
   or a lot of time with the bank) are more likely to churn compared to those that are of average tenure.
4. Worryingly, the bank is losing customers with significant bank balances which is likely to hit their 
    available capital for lending.
5. Neither the product nor the salary has a significant effect on the likelihood to churn.
'''

# finding the 1st quartile
q1 = np.quantile(df.Age, 0.25)
 
# finding the 3rd quartile
q3 = np.quantile(df.Age, 0.75)
med = np.median(df.Age)
 
# finding the iqr region
iqr = q3-q1
 
# finding upper and lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(iqr, upper_bound, lower_bound)

outliers = df.Age[(df.Age <= lower_bound) | (df.Age >= upper_bound)]
print('The following are the outliers in the boxplot:{}'.format(outliers))

# Handling age column outliers

ageNew = []
for val in df.Age:
    if val <= 60:
        ageNew.append(val)
    else:
        ageNew.append(df.Age.median())
 
df.Age = ageNew
df.Age.describe()
plt.boxplot(df.Age)
plt.show()
#### Feature engineering
## try to add features that are likely to have an impact on the probability of churning. 
## first split the train and test sets
final_dataset = pd.Series.copy(df)

# One-Hot encoding our categorical attributes
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()
##
##We need to know which the important features are. In order to find that out, 
## we are going to train the model using the Random Forest classifier.
# Finding out Feature importance using the Random Forest classifier.
# Import the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

y = final_dataset['Exited']
X = final_dataset.drop('Exited', axis = 1)
features_label = X.columns
forest = RandomForestClassifier (n_estimators = 1000, random_state = 42)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))

# Visualization of the Feature importances
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color = "green", align = "center")
plt.xticks(range(X.shape[1]), features_label, rotation = 90)
plt.show()
##  It shows the most important features are creditscore, age, tenure, balance, and so on.


### Train and build baseline model
# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Import different models 
# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
result = model_lr.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model_lr.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))
### 0.8168

###  Support Vecor Machine (SVM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
from sklearn.svm import SVC
model_svm = SVC(kernel='linear') 
model_svm.fit(X_train,y_train)
preds = model_svm.predict(X_test)
metrics.accuracy_score(y_test, preds)
print (metrics.accuracy_score(y_test, preds))
##0.7924

### Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
## 0.8688

##### Ada Boosting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators = 100, learning_rate =1)
model_ada = adb.fit(X_train,y_train)
y_pred = model_ada.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
### 0.8596

###### Gradient Boosting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
from sklearn.ensemble import GradientBoostingClassifier
gdb = GradientBoostingClassifier(n_estimators = 200, subsample =0.8)
model_gdb = gdb.fit(X_train, y_train)
y_pred = model_gdb.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print(metrics.accuracy_score(y_test, y_pred))
##0.8708

# pickling the Model
import pickle
pickle.dump(model_gdb, open("Customer_churn.pkl", "wb"))
# load the model
pickle.load(open("Customer_churn.pkl", "rb"))

