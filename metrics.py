# %%

import pandas as pd

df = pd.read_csv('./data/comunidade_dados.csv')
df.head()

# %%
df = df.replace({'Sim': 1, 'Não': 0})
df.head()

# %%

num_vars = [
    "Curte games?",
    "Curte futebol?",
    "Curte livros?",
    "Curte jogos de tabuleiro?",
    "Curte jogos de fórmula 1?",
    "Curte jogos de MMA?",
    "Idade",

]
dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)"
]

df_analysis = pd.get_dummies(df[dummy_vars]).astype(int)

df_analysis[num_vars] = df[num_vars].copy()

# %%
df_analysis['pessoa feliz?'] = df['Você se considera uma pessoa feliz?']

df_analysis

# %%
from sklearn import tree

features = df_analysis.columns[:-1].to_list()

X = df_analysis[features]
y = df_analysis['pessoa feliz?']

arvore = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=5)

arvore.fit(X,y)

# %%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analysis[['pessoa feliz?']]
df_predict['predict_arvore'] = arvore_predict

df_predict['proba_arvore'] = arvore.predict_proba(X)[:, 1]

df_predict

# %%

# Checking the df differences using mean (accuracy)
(df_predict['pessoa feliz?'] == df_predict['predict_arvore']).mean()

# %%

# Confusion Matrix
pd.crosstab(df_predict['pessoa feliz?'], df_predict['predict_arvore'])