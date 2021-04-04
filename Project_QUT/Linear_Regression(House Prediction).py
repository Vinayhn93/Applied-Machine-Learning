'''
Name: Vinay HN
Queensland University of Technology
Objective: Predict the Price of the House in Boston using Linear Regression"

Independent Variable:Xi('CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT')
 Dependent/Target Variable: Yi(Price)
 
Regression Problem: Yi is a real valued number(R)
Note: Data is too old and took in the year 1978
'''

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Loading the Data
boston = load_boston()
print("Shape of the Data :",boston.data.shape)
print("Column name : ",boston.feature_names)


bos = pd.DataFrame(boston.data)

bos['price'] = boston.target
Y = bos.price
X = bos.drop('price',axis=1)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

Lin_reg = LinearRegression()
Lin_reg.fit(X_train,Y_train)

Y_pred = Lin_reg.predict(X_test)

plt.figure(1)
plt.scatter(Y_test,Y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price Values")
plt.title("Predicting the Price of the house in Boston")
plt.show()


print('='*70)

plt.figure(2)
delt_y = Y_test - Y_pred
sns.set_style('whitegrid')
sns.kdeplot(np.array(delt_y),bw=0.5)
plt.title("PDF of the errors")
plt.show()

print('='*70)

plt.figure(3)
delt_y = Y_test - Y_pred
sns.set_style('whitegrid')
sns.kdeplot(np.array(Y_test),bw=0.5)
sns.kdeplot(np.array(Y_pred),bw=0.5)
plt.title("PDF of the Predicted(Orange) and the Actual Value(Blue)")
plt.show()

print('='*70)

print("Mean of the Actual Price of the House",np.mean(Y_test))
print("Mean of the Predicted Price of the House",np.mean(Y_pred))

'''
Conclusion:
    
Feature Engineering will reduce the Error between the actual and predicted price of the House
But It is easy to get Underfit. Hence "Regulizer is needed to Overcome the Problem of Underfir".

Note: Unfortunately we don't have the Inbuilt "Regulizer" in the SKlearn"
'''
