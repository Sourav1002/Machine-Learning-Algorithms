#Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#Dataset
data=pd.read_csv(r"C:\Users\gmode\Desktop\Data Science\Machine Learning Algorithms\Salary_Data.csv")
print(data.head())

#Data separation
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
print(data.isnull().sum())

#Data visualization
sns.barplot(x="YearsExperience",y="Salary",data=data)
plt.show()

#Test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#Predicting Result
y_pred=lr.predict(x_test)

#visualize results
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,lr.predict(x_train),color='red')
plt.title("Salary vs Experience(trainset)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,lr.predict(x_train),color='red')
plt.title("Salary vs Experience(testset)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Residuals
from sklearn import metrics
print("RMSE",np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))