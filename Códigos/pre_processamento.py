# Importando as bibliotecas e módulos necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importando o dataset
df = pd.read_csv("CSVs/fetal_health.csv")

# Entendendo o dataframe
print(f'{df.head()}\n')
print(f'{df.shape}\n')
print(f'{df.dtypes}\n')
print(f'{list(df.columns)}\n')

# Lidando com dados faltantes
print(f'{df.isna().any()}\n')

# Checando dados duplicados
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(f'{df.duplicated().sum()}\n')

# Exportando o dataframe alterado
df.to_csv("CSVs/fetal_health_manipulated.csv", index=False)

# Verificando a correlação entre as variávies do dataset
corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.0})
plt.figure(figsize=(13, 7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

# Passando as colunas que serão usadas de entrada no modelo
X = df.drop("fetal_health", axis=1)

# Passando a coluna que será usada de resposta do modelo
y = df.fetal_health

# Dividindo os conjuntos de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando os dados
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
