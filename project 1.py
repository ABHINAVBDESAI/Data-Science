#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset
df=pd.read_csv(r"C:\Users\Abhin\OneDrive\Desktop\datasets\HousingData.csv")

# Create a DataFrame from the dataset
df=df.dropna()
# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('MEDV', axis=1)  # All columns except 'MEDV' are features
y = df['MEDV']  # 'MEDV' is the target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lin_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Plot the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = (iris.target == 0).astype(int)  # Binary target: 1 if species is 'setosa', else 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# In[11]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the Adult Census Income dataset (you can download it from 'https://archive.ics.uci.edu/ml/datasets/adult')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Step 1: Load the dataset
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Step 2: Data Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Step 3: Separate features and target variable
X = data.drop('income', axis=1)  # Features
y = data['income']  # Target (<=50K or >50K)

# Step 4: Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns  # Identify categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns  # Identify numerical columns

# Build a preprocessing pipeline to handle categorical encoding and imputations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # No transformation for numerical columns
        ('cat', OneHotEncoder(), categorical_cols)  # OneHotEncode categorical columns
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a decision tree pipeline
decision_tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Step 7: Train the model
decision_tree_pipeline.fit(X_train, y_train)

# Step 8: Predict on the test set
y_pred = decision_tree_pipeline.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


# In[7]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the Adult Census Income dataset (you can download it from 'https://archive.ics.uci.edu/ml/datasets/adult')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Step 1: Load the dataset
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Step 2: Data Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Step 3: Separate features and target variable
X = data.drop('income', axis=1)  # Features
y = data['income']  # Target (<=50K or >50K)

# Step 4: Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns  # Identify categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns  # Identify numerical columns

# Build a preprocessing pipeline to handle categorical encoding and imputations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # No transformation for numerical columns
        ('cat', OneHotEncoder(), categorical_cols)  # OneHotEncode categorical columns
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a random forest pipeline
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Step 7: Train the model
random_forest_pipeline.fit(X_train, y_train)

# Step 8: Predict on the test set
y_pred = random_forest_pipeline.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


# In[7]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the Adult Census Income dataset (available at https://archive.ics.uci.edu/ml/datasets/adult)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Step 1: Load the dataset
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Step 2: Data Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Step 3: Separate features and target variable
X = data.drop('income', axis=1)  # Features
y = data['income']  # Target (<=50K or >50K)

# Step 4: Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns  # Identify categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns  # Identify numerical columns

# Build a preprocessing pipeline to handle categorical encoding and imputations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # No transformation for numerical columns
        ('cat', OneHotEncoder(), categorical_cols)  # OneHotEncode categorical columns
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create an AdaBoost pipeline with a Decision Tree as the base classifier
ada_boost_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),  # Using a simple decision tree as base estimator
        n_estimators=100,  # Number of boosting rounds
        random_state=42))
])

# Step 7: Train the model
ada_boost_pipeline.fit(X_train, y_train)

# Step 8: Predict on the test set
y_pred = ada_boost_pipeline.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


# In[12]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Step 1: Load the Wholesale Customer dataset (you can download it from https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
data = pd.read_csv(url)

# Step 2: Data Preprocessing (optional: drop categorical columns if present)
# Dropping the 'Channel' and 'Region' columns, as we will only use numerical features for clustering
X = data.drop(['Channel', 'Region'], axis=1)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Plot the Dendrogram to determine the optimal number of clusters
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Hierarchical Clustering")
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.show()

# Step 5: Apply Hierarchical Clustering (Agglomerative Clustering)
# Using 'ward' linkage to minimize variance within clusters
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# Step 6: Fit the model and predict the clusters
y_hc = hc.fit_predict(X_scaled)

# Step 7: Visualize the clusters (for simplicity, we'll only plot two features)
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[y_hc == 0, 0], X_scaled[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_hc == 1, 0], X_scaled[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_hc == 2, 0], X_scaled[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.title("Hierarchical Clustering of Wholesale Customers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# In[13]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Adult Census Income dataset (you can download it from UCI ML Repository)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'income']

# Load the dataset into a DataFrame
data = pd.read_csv(url, names=columns, na_values=' ?', sep=',', skipinitialspace=True)

# Step 2: Preprocess the data
# Drop missing values
data.dropna(inplace=True)

# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country', 'income']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop('income', axis=1)  # All features except target
y = data['income']  # Target variable (income)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA for dimensionality reduction
# Initialize PCA and fit it to the scaled data
pca = PCA(n_components=10)  # Reduce to 10 principal components (can be adjusted)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Visualize the explained variance ratio to understand how much variance each component explains
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.bar(range(1, 11), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, 11), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.title('Variance Explained by Principal Components')
plt.show()

# Step 6: Print cumulative explained variance
print(f"Cumulative explained variance by 10 components: {np.cumsum(explained_variance)[-1] * 100:.2f}%")

# Now the data `X_pca` contains the reduced dimensionality features


# In[ ]:




