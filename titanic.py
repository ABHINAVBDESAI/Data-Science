import pandas as pd
df=pd.read_csv("C:\datasets\Titanic-Dataset.csv")
print(df)
df.head()
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

lr = LogisticRegression(random_state=1)
nb = MultinomialNB()
dt = DecisionTreeClassifier(random_state = 0)
gbn = GradientBoostingClassifier(n_estimators = 10)

# le=LabelEncoder()
# le.fit(df['Age'])
# df['Age']=le.transform(df['Age'])
#
# le=LabelEncoder()
# le.fit(df['Sex'])
# df['Sex']=le.transform(df['Sex'])
#
# le=LabelEncoder()
# le.fit(df['Cabin'])
# df['Cabin']=le.transform(df['Cabin'])

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X=df.drop(['Survived','Name','Age','Sex','Ticket','Cabin','Embarked'],axis=1)
Y=df['Survived']

print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=2,test_size = 0.3)


lr.fit(X_train, Y_train)
y_pred=lr.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy :", accuracy)
