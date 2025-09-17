# %%

import pandas as pd
df = pd.read_excel('../data/dados_cerveja_nota.xlsx')
df.head()

# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']] # X is a matrix (DataFrame)
y = df['nota'] # y is a vector (Series)

# Model fitting (This is the machine learning part)
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

# %%

# Definition and presentation of coefficients
a, b = reg.intercept_, reg.coef_[0]

print(f'Intercepto: {a}, Coeficiente: {b}')

# %%

# New predictions dropping the duplicates
predict_reg = reg.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# Changing the hyperparameter max_depth to 2
arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

#Plotting the data

plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.title("Relationship between Beer and Rating")
plt.xlabel("Beer")
plt.ylabel("Rating")

# Plotting the regression line (predictions)
plt.plot(X.drop_duplicates()['cerveja'], predict_reg, color='orange')
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_full, color='green')
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_d2, color='magenta')

plt.legend(['Observations', 
            f'y = {a:.3f} + {b:.3f} x',
            'Árvore Full',
            'Árvore Depth 2'])

# %%

tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)

tree.plot_tree(arvore_full,
               feature_names=['cerveja'],
               filled=True)