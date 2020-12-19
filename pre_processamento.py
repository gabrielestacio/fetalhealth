import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotnine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fetal_health_manipulated.csv")

"""
Entendendo o dataframe

print(df.head())
print(df.shape)
print(df.dtypes)
print(list(df.columns))
"""

"""
Lidando com dados faltantes
"""

print(df.isna().any())

"""
Checando dados duplicados
"""

print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

"""
Exportando o dataframe alterado

df.to_csv("fetal_health_manipulated.csv", index=False)
"""

"""
Verificando a correlação entre as variávies do dataset

corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 4.0})
plt.figure(figsize=(13, 7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
"""

"""
Passando as colunas que serão usadas de entrada no modelo
"""

X = pd.DataFrame(columns=['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                          'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                          'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                          'percentage_of_time_with_abnormal_long_term_variability',
                          'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max',
                          'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                          'histogram_median', 'histogram_variance', 'histogram_tendency'], data=df)

"""
Passando a coluna que será usada de resposta do modelo
"""

y = pd.DataFrame(columns=['fetal_health'], data=df)

"""
Dividindo os conjuntos de treino e de teste
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Normalizando os dados
"""

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""
Verificar a possibilidade de usar PCA caso do treino demore muito
"""