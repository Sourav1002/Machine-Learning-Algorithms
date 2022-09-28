import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"kyphosis.csv")
print(data.isnull().sum())
print(data.head())

x=data.drop("Kyphosis",axis=1)
y=data["Kyphosis"]

sns.countplot(x="Age",hue="Kyphosis",data=data)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)

pred=dtree.predict(x_test)
print(pred)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print("Accuracy using Decision Tree :",accuracy_score(y_test,pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

pred1=rf.predict(x_test)
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
print("Accuracy using Decision Tree :",accuracy_score(y_test,pred1))