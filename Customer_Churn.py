import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('telcocc.csv')  # Ensure the CSV file 'telcocc.csv' is in the same directory

# Inspect the first few rows
print(df.head())

# Check for missing values and data types
df.info()

# Check the distribution of the churn column
churn_counts = df['Churn'].value_counts()
churn_counts.plot(kind='bar', title='Distribution of Churn')
plt.show()

# Plot the distribution of MonthlyCharges for churned and non-churned customers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs. Churn')
plt.show()

# Plot contract type against churn
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract Type vs. Churn')
plt.show()

# Explore churn rate by different services
services = ['InternetService', 'PhoneService', 'StreamingTV', 'StreamingMovies']
for service in services:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=service, hue='Churn', data=df)
    plt.title(f'{service} vs. Churn')
    plt.show()

# Distribution of tenure with respect to churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs. Churn')
plt.show()

# Convert 'TotalCharges' to numeric, setting non-numeric values to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Optionally handle missing values (fill with mean or drop)
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Correlation matrix for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Gender vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('Gender vs. Churn')
plt.show()

# Load the dataset
file_path=file_path = 'telcocc.csv'
telco_data = pd.read_csv(file_path)


# Display the first few rows and general info to understand the dataset
telco_data.head(), telco_data.info()

# Set up visual style for plots
sns.set(style="whitegrid")
plt.figure(figsize=(16, 20))

# Set up visual style for plots
sns.set(style="whitegrid")
plt.figure(figsize=(16, 24))

# 1. Bivariate Analysis of Demographic Features vs. Churn
demographic_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
for i, feature in enumerate(demographic_features, 1):
    plt.subplot(4, 2, i)
    sns.countplot(data=telco_data, x=feature, hue="Churn", palette="pastel")
    plt.title(f'Distribution of {feature} by Churn', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)

plt.tight_layout(pad=3.0)
plt.show()

# 2. Bivariate Analysis of Service Features vs. Churn
plt.figure(figsize=(16, 28))
service_features = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i, feature in enumerate(service_features, 1):
    plt.subplot(5, 2, i)
    sns.countplot(data=telco_data, x=feature, hue="Churn", palette="muted")
    plt.title(f'Distribution of {feature} by Churn', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)

plt.tight_layout(pad=3.0)
plt.show()

# 3. Bivariate Analysis of Billing Features vs. Churn
plt.figure(figsize=(16, 16))

# Contract type vs Churn
plt.subplot(3, 1, 1)
sns.countplot(data=telco_data, x="Contract", hue="Churn", palette="cool")
plt.title("Distribution of Contract Type by Churn", fontsize=14)
plt.xlabel("Contract", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)

# Paperless Billing vs Churn
plt.subplot(3, 1, 2)
sns.countplot(data=telco_data, x="PaperlessBilling", hue="Churn", palette="cool")
plt.title("Distribution of Paperless Billing by Churn", fontsize=14)
plt.xlabel("Paperless Billing", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)

# Payment Method vs Churn
plt.subplot(3, 1, 3)
sns.countplot(data=telco_data, x="PaymentMethod", hue="Churn", palette="cool")
plt.title("Distribution of Payment Method by Churn", fontsize=14)
plt.xlabel("Payment Method", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout(pad=3.0)
plt.show()

# 4. Continuous Variables vs. Churn
# Tenure vs Churn
plt.figure(figsize=(12, 6))
sns.boxplot(data=telco_data, x="Churn", y="tenure", palette="Set2")
plt.title("Distribution of Tenure by Churn", fontsize=14)
plt.xlabel("Churn", fontsize=12)
plt.ylabel("Tenure (months)", fontsize=12)
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(12, 6))
sns.boxplot(data=telco_data, x="Churn", y="MonthlyCharges", palette="Set2")
plt.title("Distribution of Monthly Charges by Churn", fontsize=14)
plt.xlabel("Churn", fontsize=12)
plt.ylabel("Monthly Charges", fontsize=12)
plt.show()

# Total Charges vs Churn
plt.figure(figsize=(12, 6))
sns.boxplot(data=telco_data, x="Churn", y="TotalCharges", palette="Set2")
plt.title("Distribution of Total Charges by Churn", fontsize=14)
plt.xlabel("Churn", fontsize=12)
plt.ylabel("Total Charges", fontsize=12)
plt.show()

# Data Cleaning: Convert 'TotalCharges' to numeric, setting non-numeric values to NaN, and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Encoding categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardize the features for better model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predictions and evaluation
y_pred = logreg.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.Series(logreg.coef_[0], index=df.drop('Churn', axis=1).columns)
feature_importance = feature_importance.sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Feature Importance in Predicting Churn")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()

# Interaction Plot (e.g., Monthly Charges and Contract Type)
# Reload data for clarity in interaction plots, keep categorical variables as original labels
df_interaction = pd.read_csv('telcocc.csv')
plt.figure(figsize=(12, 6))
sns.boxplot(x='Contract', y='MonthlyCharges', hue='Churn', data=df_interaction)
plt.title("Interaction between Monthly Charges and Contract Type on Churn")
plt.show()
