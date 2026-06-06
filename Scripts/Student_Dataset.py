import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv(r'c:\Users\User\OneDrive\Desktop\synthetic_student_dataset_missing.csv')

#display first 5 rows
print(df.head())

#display shape
print(df.shape)

#display info
print(df.info())

#display summary statistics
print(df.describe())

#check missing values
print(df.isnull().sum())

#numerical columns cleaning
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#categorical values cleaning
cat_cols= df.select_dtypes(include=['object']).columns
for cols in cat_cols:
    df[cols] = df[cols].fillna(df[cols].mode()[0])

#feature engineering
#creating new feature 'average score'
df['Average_Score'] = (df['math_score'] + df["science_score"] + df['english_score']) / 3

#creating new features 'Total score'
df['Total_Score'] = df['math_score'] + df['science_score'] + df['english_score']

#creating new feature 'pass / fail' based on average score
df['Result'] = df['Average_Score'].apply(lambda x: 'pass' if x>= 40 else'fail')

#label encoding of pass / fail column
df['Result_encoded'] = df['Result'].replace({'pass':1, 'fail':0})

#saved cleaned dataset
df.to_csv(r'c:\Users\User\OneDrive\Desktop\cleaned_student_dataset.csv', index=False)

#Distribution of average score
plt.figure(figsize=(10,6))
sns.histplot(df["Average_Score"], bins = 20, kde = True)
plt.title("Distribution of Average Score")
plt.show()

#Study hours vs average score
plt.figure(figsize=(10,6))
sns.scatterplot(x='study_hours', y='Average_Score', data=df, hue="Result")
plt.title("Study Hours vs Average Score")
plt.show()

#average score by gender
plt.figure(figsize=(10,6))
sns.barplot(x = 'gender', y = 'Average_Score', data=df)
plt.title("Average Score by Gender")
plt.show()

#Pass / Fail distribution
plt.figure(figsize=(10,6))
sns.countplot(x='Result', data=df)
plt.title("Pass vs Fail Distribution")
plt.show()

#correlation heatmap
corr_matrix = df.corr(numeric_only = True)
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot = True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()