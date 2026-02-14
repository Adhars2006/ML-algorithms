import pandas as pd
import numpy as np

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Age'] = pd.cut(df['Age'],
                   bins=[0, 12, 20, 40, 60, 100],
                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

df['SibSp'] = df['SibSp'].apply(lambda x: 'Yes' if x > 0 else 'No')
df['Parch'] = df['Parch'].apply(lambda x: 'Yes' if x > 0 else 'No')
df['Pclass'] = df['Pclass'].astype(str)

# Optional: use small subset to avoid too much generalization
df = df.head(20)

X = df.drop('Survived', axis=1).values
y = df['Survived'].values

# -----------------------------
# Candidate Elimination
# -----------------------------
def candidate_elimination(X, y):

    S = ['0'] * len(X[0])
    G = [['?'] * len(X[0])]

    for i in range(len(X)):
        if y[i] == 1:  # Positive example
            for j in range(len(X[0])):
                if S[j] == '0':
                    S[j] = X[i][j]
                elif S[j] != X[i][j]:
                    S[j] = '?'

            G = [g for g in G if all(g[k] == '?' or g[k] == S[k] for k in range(len(S)))]

        else:  # Negative example
            new_G = []
            for g in G:
                for j in range(len(X[0])):
                    if g[j] == '?':
                        if S[j] != '?':
                            new_hypothesis = g.copy()
                            new_hypothesis[j] = S[j]
                            new_G.append(new_hypothesis)
            G = new_G if new_G else G

    return S, G

S_final, G_final = candidate_elimination(X, y)

# -----------------------------
# Output
# -----------------------------
print("Final Specific Boundary (S):")
print(S_final)

print("\nFinal General Boundary (G):")
for g in G_final:
    print(g)
