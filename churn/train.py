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
#Sample (S)

from sklearn import  model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
# Stratify is used to maintain the same proportion of the target variable in both datasets


print("Taxa variável resposta - Original:", y.mean())
print("Taxa variável resposta - Treino:", y_train.mean())
print("Taxa variável resposta - Teste:", y_test.mean())

# %%
# EDA (E) - using only the training dataset

# Checking for missing values
X_train.isna().sum().sort_values(ascending=False)

# %%
# Bivariate analysis
df_analysis = X_train.copy()
df_analysis[target] = y_train
summary = df_analysis.groupby(by=target).agg(['mean', 'median']).T
summary

# %%
summary['diff_abs'] = summary[0] - summary[1]
summary['diff_rel'] = summary[0] / summary[1]
summary.sort_values(by='diff_rel', ascending=False)

# %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# %%
feature_importances = pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()

feature_importances['acum'] = feature_importances[0].cumsum()
# feature_importances[feature_importances[0] > 0.01]
feature_importances[feature_importances['acum'] < 0.96]

# %%
