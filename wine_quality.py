import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

lr = RandomForestClassifier(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 0)
gbn = GradientBoostingClassifier(n_estimators = 10)
df = pd.read_csv("D:\Tejal\winequalityN.csv")
#print(df)

df['fixed acidity'].fillna((df['fixed acidity']).mean(), inplace = True)
df['volatile acidity'].fillna((df['volatile acidity']).mean(), inplace = True)
df['citric acid'].fillna((df['citric acid']).mean(), inplace = True)
df['residual sugar'].fillna((df['residual sugar']).mean(), inplace = True)
df['chlorides'].fillna((df['chlorides']).mean(), inplace = True)
df['pH'].fillna((df['pH']).mean(), inplace = True)
df['sulphates'].fillna((df['sulphates']).mean(), inplace = True)
X = df.drop('type', axis = 1)
X = X.drop('sulphates', axis =1)
X = X.drop('volatile acidity', axis =1)
X = X.drop('fixed acidity', axis =1)
X = X.drop('citric acid', axis =1)
Y = df['quality']

print(df.isnull().sum())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=2,test_size = 0.3)


lr.fit(X_train, Y_train)
y_pred=lr.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)









