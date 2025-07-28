# %%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./data/dados_cerveja_nota.xlsx')

df.head()

df['aprovado'] = (df['nota'] > 5).astype(int)

# %%
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')

# %%

from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)

reg.fit(df[['cerveja']], df['aprovado'])

reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_predict

plt.plot(df[['cerveja']].drop_duplicates(),
         reg_predict, 'o', color='orange')

# %%

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(),
         reg_predict, color='orange')


# %%

reg_prob = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), reg_prob, color='red')
plt.hlines(0.5,xmin=1, xmax=9, linestyles='--', color='black')

plt.legend(['Observation', 'Reg Prediction', 'Reg Probability'])

# %%
from sklearn import tree

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])

arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_prob = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), reg_prob, color='red')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_predict, color='green')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_prob, color='magenta')

plt.hlines(0.5,xmin=1, xmax=9, linestyles='--', color='black')

plt.legend(['Observation', 'Reg Prediction', 'Reg Probability', 'Tree Prediction', 'Tree Probability'])

# %%

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])

arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_prob = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), reg_prob, color='red')

plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict, color='blue')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_prob, color='gray')

plt.hlines(0.5,xmin=1, xmax=9, linestyles='--', color='black')

plt.legend(['Observation', 'Reg Prediction', 'Reg Probability', 'Tree D2 Prediction', 'Tree D2 Probability'])


# %%

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])

nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_prob = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), reg_prob, color='red')

plt.plot(df[['cerveja']].drop_duplicates(), nb_predict, color='green')
plt.plot(df[['cerveja']].drop_duplicates(), nb_prob, color='magenta')

plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict, color='blue')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_prob, color='gray')

plt.hlines(0.5,xmin=1, xmax=9, linestyles='--', color='black')

plt.legend(['Observation', 'Reg Prediction', 'Reg Probability', 'NB Prediction', 'NB Probability', 'Tree D2 Prediction', 'Tree D2 Probability'])

# %%

# Comparison of Regression and Naive Bayes

plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), reg_prob, color='red')

plt.plot(df[['cerveja']].drop_duplicates(), nb_predict, color='green')
plt.plot(df[['cerveja']].drop_duplicates(), nb_prob, color='magenta')

plt.hlines(0.5,xmin=1, xmax=9, linestyles='--', color='black')

plt.legend(['Observation', 'Reg Prediction', 'Reg Probability', 'NB Prediction', 'NB Probability'])