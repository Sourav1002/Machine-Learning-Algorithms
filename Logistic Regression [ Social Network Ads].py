import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"SocialNetworkAds.csv")
print(data.isnull().sum())

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

sns.heatmap(data.corr())
plt.show()
#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#logsitic regression and confusion matrix
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)

#accuracy
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred))
