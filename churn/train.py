# %%
import pandas as pd

df = pd.read_csv("../data/abt_churn.csv")
df.head()

# %%
# Out of Time (OOT) dataset - data from the most recent reference date
oot = df[df['dtRef']==df['dtRef'].max()].copy()
oot

# %%
# Split the rest of the data into training and validation datasets
df_train = df[df['dtRef']<df['dtRef'].max()].copy()
df_train['dtRef'].max()


# %%
# Defining the features and target
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%
from sklearn import  model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
# Stratify is used to maintain the same proportion of the target variable in both datasets

# %%
print("Taxa variável resposta - Original:", y.mean())
print("Taxa variável resposta - Treino:", y_train.mean())
print("Taxa variável resposta - Teste:", y_test.mean())

# %%
