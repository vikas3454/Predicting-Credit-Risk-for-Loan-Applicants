import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv('german_credit_data_with_labels copy2.csv')

print(df.info())
print(df.describe())

print(df.isnull().sum())


df.hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 2. Handling Missing Values
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')  
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])


categorical_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')  
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# 3. Handling Outliers (optional)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 4. Feature Engineering (Example)


df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])

df['Credit_to_Income'] = df['Credit amount'] / df['Income']  # assuming 'Income' is a column in your dataset

# 5. Data Scaling (optional)
scaler = StandardScaler()
scaled_numerical_cols = df[numerical_cols]
df[numerical_cols] = scaler.fit_transform(scaled_numerical_cols)

# 6. Splitting Data into Features and Target


X = df.drop(columns=['target']) 
y = df['target']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of the cleaned and preprocessed data
print(df.head())
