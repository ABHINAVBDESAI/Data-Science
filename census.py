import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 0)
gbn = GradientBoostingClassifier(n_estimators = 10)
df = pd.read_csv("D:\Tejal\census.csv")
#print(df)

X = df.drop('age', axis = 1)
X = X.drop('workclass', axis =1)
X = X.drop('relationship', axis =1)
X = X.drop('fnlwgt', axis =1)
X = X.drop('marital.status', axis =1)
X = X.drop('education', axis =1)
X = X.drop('occupation', axis =1)
X = X.drop('race', axis =1)
X = X.drop('sex', axis =1)
X = X.drop('native.country', axis =1)
X = X.drop('income', axis =1)
Y = df['income']
#print(df.isnull().sum())


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=2,test_size = 0.3)


lr.fit(X_train, Y_train)
y_pred=lr.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)
