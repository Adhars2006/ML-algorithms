import pandas as pd
import numpy as np

df = pd.read_csv("Titanic-Dataset.csv")  

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Age'] = pd.cut(df['Age'],
                   bins=[0, 12, 20, 40, 60, 100],
                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

df['SibSp'] = df['SibSp'].apply(lambda x: 'Yes' if x > 0 else 'No')
df['Parch'] = df['Parch'].apply(lambda x: 'Yes' if x > 0 else 'No')
df['Pclass'] = df['Pclass'].astype(str)

X = df.drop('Survived', axis=1).values
y = df['Survived'].values


def find_s(X, y):
    hypothesis = ['0'] * len(X[0])

    for i in range(len(X)):
        if y[i] == 1:  
            for j in range(len(X[0])):
                if hypothesis[j] == '0':
                    hypothesis[j] = X[i][j]
                elif hypothesis[j] != X[i][j]:
                    hypothesis[j] = '?'
    
    return hypothesis

final_hypothesis = find_s(X, y)

print("Final Hypothesis from Find-S:")
for feature, value in zip(df.drop('Survived', axis=1).columns, final_hypothesis):
    print(f"{feature} : {value}")
