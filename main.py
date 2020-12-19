import pandas as pd
import numpy as np
# import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
# import plotnine


df = pd.read_csv("fetal_health_manipulated.csv")

"""
Entendendo o dataframe
"""

print(df.head())
print(df.shape)
print(df.dtypes)
print(list(df.columns))

"""
Lidando com dados faltantes
"""

print(df.isna().any())

"""
Checando dados duplicados
"""

print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
print(df.shape)

"""
Exportando o dataframe alterado
"""

df.to_csv("fetal_health_manipulated.csv", index=False)

"""
Verificando a correlação entre as variávies do dataset
"""

corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 4.0})
plt.figure(figsize=(13, 7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
