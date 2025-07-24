# %%

import pandas as pd
df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df.head()

# %%

from sklearn import linear_model

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
predict = reg.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

#Plotting the data

plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.title("Relationship between Beer and Rating")
plt.xlabel("Beer")
plt.ylabel("Rating")

# Plotting the regression line (predictions)
plt.plot(X.drop_duplicates()['cerveja'], predict)

plt.legend(['Observations', f'y = {a:.3f} + {b:.3f} x'])