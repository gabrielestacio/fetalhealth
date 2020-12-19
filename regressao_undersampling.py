import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotnine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

df = pd.read_csv("fetal_health_manipulated.csv")

print(df.isna().any())

print(df.duplicated().sum())
df.drop_duplicates(inplace=True)

X = pd.DataFrame(columns=['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                          'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                          'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                          'percentage_of_time_with_abnormal_long_term_variability',
                          'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max',
                          'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
                          'histogram_median', 'histogram_variance', 'histogram_tendency'], data=df)

y = pd.DataFrame(columns=['fetal_health'], data=df)

nr = NearMiss()
X, y = nr.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print('\n')

"""
Treinamento
"""

logit = LogisticRegression(verbose=1, max_iter=1000)
logit.fit(X_train, np.ravel(y_train, order='C'))
y_pred = logit.predict(X_test)

# Score de acurácia
print(metrics.accuracy_score(y_test, y_pred))

"""
Métricas
"""


"""
Balanceamento das classes
"""


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(df['fetal_health'].value_counts())
print(metrics.classification_report(y_test, y_pred, target_names=['1', '2', '3']))

# ____________________________________________________________________________________________________________
