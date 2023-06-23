import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("D:\Tejal\wal.csv")
print(df)
print(df.info())

df['year']=df['Date'].apply(lambda x:x[:4])
df['month']=df['Date'].apply(lambda x:x[5:7])
df=df.drop('Date',axis=1)
print(df)

Y=df['Weekly_Sales'].copy()
X=df.drop('Weekly_Sales',axis=1).copy()


scaler=StandardScaler()
X=scaler.fit_transform(X)


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2,test_size=0.3)


lr=LogisticRegression()
S=SVC()
dtc=DecisionTreeClassifier()
lr.fit(X_train,Y_train)
S.fit(X_train,Y_train)
dtc.fit(X_train,Y_train)

print("Logistic Regression Accuracy : ",lr.score(X_test,Y_test))
print("SVM Accuracy : ",S.score(X_test,Y_test))
print("Decision Tree Accuracy : ",dtc.score(X_test,Y_test))









