import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"SocialNetworkAds.csv")
data.head()
print(data.isnull().sum())

x=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

sns.heatmap(data.corr())
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print("Accuracy :",accuracy_score(y_test,y_pred))