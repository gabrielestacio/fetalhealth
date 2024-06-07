# **Fetal Health**
Olá! Essa é a documentação do projeto Fetal Health, desenvolvido por Gabriel Estácio de Souza Passos para o processo seletivo de bolsistas para o projeto "PESQUISA APLICADA PARA INTEGRAÇÃO INTELIGENTE ORIENTADA AO FORTALECIMENTO DAS REDES DE ATENÇÃO PARA RESPOSTA RÁPIDA À SÍFILIS", do LAIS/HUOL. A base de dados utilizada para o desenvolvimento deste modelo pode ser encontrada [aqui](https://www.kaggle.com/andrewmvd/fetal-health-classification). 
<br/><br/>

## **Partes da Documentação**
[1.   Apresentação do problema](https://colab.research.google.com/drive/1yyZYx8dbw5oPXg2G3gr8Ky3XzItKscfN#scrollTo=SkQsNphyBROJ&line=6&uniqifier=1)

[2.   Metodologia e abordagem](https://colab.research.google.com/drive/1yyZYx8dbw5oPXg2G3gr8Ky3XzItKscfN#scrollTo=gjLNV1fiBLFM&line=4&uniqifier=1)

[3.   Aquisição e representação do conhecimento](https://colab.research.google.com/drive/1yyZYx8dbw5oPXg2G3gr8Ky3XzItKscfN#scrollTo=vyr8ft0-reYf&line=6&uniqifier=1)

[4.   Códigos e tutorial do projeto](https://colab.research.google.com/drive/1yyZYx8dbw5oPXg2G3gr8Ky3XzItKscfN#scrollTo=JO1MpvBF1Jt5&line=1&uniqifier=1ttps://)
<br/><br/>

## **Apresentação do Problema**
O problema consiste na análise de dados do exame cardiotocogramas (CTGs) que tem por objetivo medir a frequência cardíaca fetal (FCF), movimentos fetais, contrações uterinas e outros parâmetros, a fim de prevenir a mortalidade infantil e materna. O conjunto de dados contém 2.126 registros, que foram classificados por três obstetricistas especialistas em 3 classes: Normal (1), Suspeito (2) e Patológico (3). Sendo assim, o objetivo é criar um modelo multiclasse que classifique os dados nestes três estados de saúde fetal.
<br/><br/>

## **Metodologia e Abordagem**
Para resolver este problema, utilizaremos a linguagem de programação **Python**, na versão 3.8, com as bibliotecas **pandas**, **numpy**, **matplotlib**, **seaborn** e **scikit-learn**.
Nosso modelo tem como domínio do conhecimento a saúde fetal, através da cardiotocografia, usada para medir o bem-estar fetal. O conjunto de labels é formado pelos dados da coluna ***'fetal_health'*** do conjunto de dados fornecido. As características dos dados usados para o treinamento são:
*   Frequência cardíaca fetal;
*   Número de acelerações por segundo;
*   Movimentos fetais por segundo;
*   Contrações uterinas por segundo;
*   Desacelerações leves por segundo;
*   Desacelerações prolongadas por segundo;
*   Porcentagem de tempo com uma variabilidade de curto prazo anormal;
*   Valor médio da variabilidade de curto prazo;
*   Porcentagem de tempo com uma variabilidade de longo prazo anormal;
*   Valor médio da variabilidade de longo prazo;
*   Atributos do histograma cardíaco: largura, valores máximos e míninmos, número de picos e de zeros, média, variância e tendência.

A saída desejada é um número de 1 a 3, que indicará o estado de saúde do feto: normal, suspeito ou patológico, respectivamente, utilizando um modelo preditivo multiclasse com aprendizado supervisionado através da técnica de regressão logística para geração de conhecimento.

A técnica de regressão logística foi a escolhida pois esta dinâmica trabalha retornando valores entre 0 e 1, o que é bastante útil no nosso problema, onde devemos gerar uma classificação do estado de saúde do feto entre saudável, ter suspeita de uma patologia, ou efetivamente ter uma patologia.  A ideia é que, a partir dos dados de entrada, pela regressão logística, o modelo calcule a probabilidade do feto estar saudável ou não, e a partir desse resultado, categorize este feto em uma das três classes.

Essa dinâmica é a mais comum quando trabalhamos com modelos cujas predições são valores de variáveis categóricas tomados a partir de uma série de variáveis contínuas ou binárias. É comumente utilizada em problemas que envolvem saúde, pois permite criar um modelo que caracterize indivíduos, como normal, supeito ou patológico, no nosso caso, gerando o modelo multiclasse esperado.

Matematicamente, a regressão logística usa a função sigmóide (conhecida como logit) para calcular uma função discriminante que descreva a probabilidade à posteriori de um modelo.
<br/><br/>

## **Aquisição e Representação do Conhecimento**

**PROCESSO DE AQUISIÇÃO DO CONHECIMENTO**

O processo de aquisição do conhecimento passou pelas seguintes etapas:

**1.   Entendimento do Domínio - Etapa Manual:**
Baseado em entrevistas, análises e descrições, através da consulta com especialistas da área, da observação do dataset fornecido e das fontes bibliográficas abaixo:


*   [Silveira SK, Trapani Júnior A.
Monitorização fetal intraparto.
](https://docs.bvsalud.org/biblioref/2020/03/1052446/femina-2019-481-59-64.pdf) São Paulo: Federação Brasileira
das Associações de Ginecologia
e Obstetrícia (Febrasgo); 2018.
(Protocolo Febrasgo – Obstetrícia,
nº 100/Comissão Nacional
Especializada em Assistência ao
Abortamento, Parto e Puerpério);

*   [Oliveira CA, Sá RA. Cardiotocografia
anteparto.](https://docs.bvsalud.org/biblioref/2020/06/1099676/femina-2020-485-316-320.pdf) São Paulo: Federação
Brasileira das Associações de
Ginecologia e Obstetrícia (Febrasgo).  2018. (Protocolo Febrasgo –
Obstetrícia, nº 81/Comissão
Nacional Especializada em Medicina
Fetal);

*   [Artigo sobre Cardiotocografia](https://en.wikipedia.org/wiki/Cardiotocography#Periodic_or_episodic_decelerations), do Wikipedia.

**2.   Definindo o Problema - Etapa Manual:**
O estado inicial desse problema é de interpretação manual dos dados para determinar a situação da saúde do feto. Este processo pode acarretar em erros se algum dos parâmetros for interpretado incorretamente. Em cima disso, criaremos um modelo de aprendizado de máquina a fim de automatizar este processo, treinando-o para classificar cada entrada com cada vez mais precisão. O problema pode ser considerado resolvido quando as métricas retornarem uma boa avaliação do modelo.

**3.   Aprendizado de Máquina - Etapa Automática:**
Fizemos a atribuição de um conjunto de dados que foram utilizados para construir a base de conhecimento do nosso modelo. A partir disso, o modelo foi treinado para executar seu objetivo.

**PROCESSO DE REPRESENTAÇÃO DO CONHECIMENTO:**

O processo de representação do conhecimento foi feito através de frames, como mostra a imagem abaixo:

![Representação do Conhecimento](https://github.com/gabrielestacio/fetalhealth/assets/53371887/527e873d-b5b6-472a-ac94-72dcdd6d98af)

## **Tutorial do Projeto**
**1: PRÉ-PROCESSAMENTO**

Importação, análise e preparação dos dados para o treinamento e teste do modelo.

Inicialmente, iremos fazer o **import** das bibliotecas e módulos que usaremos nesse projeto. Serão usados métodos das bibliotecas ***pandas***, ***numpy***, ***scikit-lear (sklearn)***, ***matplotlib*** e ***seaborn***.
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
```

O próximo passo é importar o dataset fornecido para o projeto, por meio da função ***read_csv()*** da biblioteca ***pandas***:
```Python
df = pd.read_csv("fetal_health.csv")
```

Pra que entendamos melhor do que se trata o nosso dataset, podemos fazer algumas análises sobre ele. Iniciaremos verificando algumas características do conjunto.
```Python
# imprimindo as 20 primeiras linhas do dataframe
df.head(20)
```
```Python
# imprimindo o formato do dataframe (linhas x colunas)
df.shape
```
```Python
# imprimindo os tipos de dados contidos em cada coluna
df.dtypes
```
```Python
# listando o nome de todas as colunas (atributos)
list(df.columns)
```

Pela função ***df.head()***, podemos observar uma tendência da coluna **'severe_decelerations'** de ter valores muito parecidos. Vejamos se essa tendência se confirma.
```Python
df['severe_decelerations'].value_counts()
```

Esta coluna realmente tem valores quase que constantes. Portanto, vamos excluí-la do dataframe por meio da função ***drop()***, da bibliioteca ***pandas***.
```Python
df.drop('severe_decelerations', axis=1, inplace=True)
```

Agora, iremos analisar se este dataset possui dados faltantes. Para isso, juntaremos as funções ***isna()***, que verifica se um elemento é nulo, e ***any()***, que retorna 'True' ou 'False' pra se a condição da função anterior foi satisfeita em cada coluna. As duas pertencem a biblioteca ***pandas***.
```Python
# retornando se alguma das colunas tem dados ausentes
df.isna().any()
```

Ótimo! Nosso dataframe está todo completo e não possui dados ausentes. Sendo assim, podemos passar para o próximo passo, que é verificar se temos dados duplicados através da junção das funções ***duplicated()*** e ***sum()***, da biblioteca ***pandas***.
```Python
# retornando a quantidade de dados duplicados que temos em todo nosso dataset
df.duplicated().sum()
```

Descobrimos que temos alguns dados duplicados, então precisamos tratá-los. Para isso, usaremos a função ***drop_duplicates()***:
```Python
# eliminando dados duplicados
df.drop_duplicates(inplace=True)
```

E agora, se repetirmos o código anterior, veremos que esses dados duplicados se foram.
```Python
# retornando a quantidade de dados duplicados que temos em todo nosso dataset
print(df.duplicated().sum())
```

A próxima etapa do pré-processamento é verificar se temos variáveis muito correlacionadas e que podem ser unidas. Verificaremos isso fazendo um mapa de calor, usando os seguintes métodos:


*   **pandas**: ***corr()***

*   **numpy**: ***zeros_like()*** e ***triu_indices_from()***

*   **seaborn**: ***set_context()***, ***heatmap()***, ***set_xticklabels()***, ***get_xticklabels()***, ***set_yticklabels()*** e ***get_yticklabels()***

*   **matplotlib**: ***figure()***

```Python
# criando um mapa de calor que mostra a correlação entre as variáveis

corr = df.corr()
sns.set_context("notebook", font_scale=0.8, rc={"lines.linewidth": 4.0})
plt.figure(figsize=(13, 7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

"""
Lembrando que a diagonal principal representa a correlação de uma variável
consigo mesma, por isso todas as posições têm valor '1.00'
"""
```

Como podemos perceber pelo gráfico, as variáveis de média, moda e mediana do histograma cardíaco tem uma correlação muito alta. Portanto, iremos excluir as colunas de ***'histogram_mode'*** e ***'histogram_median'***, permanecendo com a coluna ***histogram_mean***. Faremos isso usando a função ***drop()***, da bibliioteca ***pandas***.
```Python
df.drop(columns=['histogram_mode', 'histogram_median'], axis=1, inplace=True)
```

Como vimos na [célula de checagem de tipos](https://colab.research.google.com/drive/1yyZYx8dbw5oPXg2G3gr8Ky3XzItKscfN#scrollTo=K-_kceMDjRx-&line=2&uniqifier=1), temos apenas váriaveis numéricas, então não precisamos fazer conversões de variáveis categóricas, concluindo a limpeza dos nossos dados. Por segurança, exportaremos esse dataframe já limpo, para não precisarmos repetir todo o processo numa necessidade futura. Faremos isso usando a função ***to_csv()*** da biblioteca ***pandas***.
```Python
# exportando o dataframe manipulado
df.to_csv("fetal_health_manipulated.csv", index=False)
```

Nosso próximo passo é determinar nossos conjuntos de entradas e de respostas. Utilizaremos a função ***drop()*** da biblioteca ***pandas*** para criar o conjunto de entradas, utilizando as colunas que caracterizam nosso grupo de treinamento. Apesar de ser uma função usada para apagar dados, nós não a utilizaremos pra modificar nosso dataframe. Na prática, iremos armazená-lo em uma outra variável, excluindo apenas nossa coluna de respostas, sem mudar o que está armazenado em '**df**'.
```Python
"""
Armazenando as colunas que serão usadas de entrada no nosso modelo, ou seja,
todas as colunas, menos a de resposta
"""
X = df.drop("fetal_health", axis=1)
```

Para o conjunto de labels (respostas), armazenaremos em uma nova variável a coluna que "removemos" na linha anterior
```Python
y = df.fetal_health
```

A penúltima etapa do pré-processamento é criar a validação cruzada. Usaremos o método holdout, dividindo nossos conjuntos em outros conjuntos de treinamento e de teste, através da função train_test_split(), da biblioteca scikit-learn, na seguinte proporção: 70% para treino do modelo e 30% para teste.
```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Antes de finalizarmos o pré-processamento, podemos notar que o valor mínimo e o valor máximo de algumas váriaveis de entrada são bem distintos:
```Python
# exibindo as 20 primeiras linhas de cada coluna
df.head(20)
```

Váriaveis como **fetal_movement**, **abnormal_short_term_variability** e **mean_value_of_short_term_variability** tem valores bem distintos, dentro de suas escalas. Pra resolvermos isso, vamos normalizar nossos dados utilizando as funções ***StandardScaler()***, ***fit_transform()*** e ***transform()*** da biblioteca ***scikit-learn***.
```Python
# normalizando os dados
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

Após esse passo, terminamos o pré-processamento. Agora, iremos iniciar o treinamento do nosso modelo.

**2: TREINAMENTO E TESTE DO MODELO**

Aplicação da dinâmica de treinamento para aprimoramento do modelo.

Para o treinamento do nosso modelo, utilizaremos a função ***ravel()*** da biblioteca ***numpy***, e as funções ***LogisticRegression()***, ***fit()*** e ***predict()***, da biblioteca ***scikit-learn***.
```Python
# fazendo o treinamento por regressão logística e armazenando o resultado das predições em uma nova variável
logit = LogisticRegression(verbose=1, max_iter=1000)
logit.fit(X_train, np.ravel(y_train, order='C'))
y_pred = logit.predict(X_test)
```

Finalizado o treinamento do nosso modelo, podemos prosseguir para a última etapa: a apresentação das métricas e avaliação do modelo.

**3: MÉTRICAS**

Apresentação das estatísticas relacionadas ao treinamento e avaliação do modelo.

Iniciaremos verificando a acurácia do nosso modelo através da função ***score()***, da biblioteca ***scikit-learn***.
```Python
# Acurácia
'''
soma dos positivos dividido pela soma de positivos e negativos
'''
print(f'Acurácia:\n{logit.score(X_test, y_test)}')
```

Nosso modelo obteve uma acurácia de, aproximadamente, 91.6%. Vamos verificar como isso se traduziu nas classes 1 (normal), 2 (suspeito) e 3 (patológico), através de um relatório de classificação. Para isso, exibiremos as métricas de matriz de confusão, sensibilidade, precisão, especificidade e f1_score .

Para a matriz de confusão, usaremos o métod ***confusion_matrix()***, do módulo ***metrics*** da biblioteca ***scikit-learn***.
```Python
# Matriz de confusão
'''
retorna os positivos e negativos de cada classe em formato de matriz
'''
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(f'Matriz de confusão:\n{cnf_matrix}\n')
```

Para as outras métricas, utilizaremos apenas o resultado da matriz de confusão para calcular os verdadeiros (positivo e negativo) e os falsos (positivo e negativo). Além disso, usaremos a função ***append()*** e ***sum()***, que são funções *built-in* do Python, para adicionar elementos nas listas e somar todos os elementos de uma lista, respectivamente.
```Python
# Sensibilidade
'''
snes = lista com sensibilidade de cada classe
ssens = soma de falsos negativos e verdadeiros positivos

verdadeiros positivos divididos pela soma dos verdadeiros positivos e
falsos negativos
'''

sens = []
for i in range(3):
    ssens = sum(cnf_matrix[i])
    sens.append(cnf_matrix[i][i]/ssens)
    print(f'Sensibilidade (recall) de {i+1}: {sens[i]}')
```
```Python
# Especificidade
'''
espec = lista com especificidade de cada classe
fpos = numero de falsos positivos
vneg = numero de verdadeiros negativos

verdadeiros negativos divididos pela soma de verdadeiros negativos e falsos
positivos
'''

espec = []
fpos = 0
vneg = 0
for i in range(3):
    for j in range(3):
        fpos += cnf_matrix[j][i]
    fpos -= cnf_matrix[i][i]
    vneg = sum(sum(cnf_matrix)) - sum(cnf_matrix[i]) - fpos
    espec.append(vneg/(vneg+fpos))
    print(f'Especificidade de {i+1}: {espec[i]}')
    fpos = 0
```
```Python
# Precisão
'''
prec = lista com precisão de cada classe
fpos = numero de falsos positivos
vpos = numero de verdadeiros positivos

verdadeiros positivos dividido pela soma dos positivos
'''

prec = []
fpos = 0
for i in range(3):
    for j in range(3):
        fpos += cnf_matrix[j][i]
    fpos -= cnf_matrix[i][i]
    vpos = cnf_matrix[i][i]
    prec.append(vpos/(vpos+fpos))
    print(f'Precisão de {i+1}: {prec[i]}')
    fpos = 0
```
```Python
# f1_score
'''
dobro do produto da precisão com a sensibilidade divido pela soma da precisão com a sensibilidade
'''

f1_score = []
for i in range(3):
    f1_score.append((2*prec[i]*sens[i])/(prec[i]+sens[i]))
    print(f'f1_score de {i + 1}: {f1_score[i]}')
```

Podemos observar, através da precisão e da sensibilidade que, para as classes 1 e 3, o modelo se comportou bem na previsão de verdadeiros e teve um alto índice de acertos, contribuindo para o f1_score alto. Já a classe 2, não teve um desempenho muito bom, principalmente na classificação de verdadeiros positivos, o que impactou na sensibilidade, pois o modelo classificou muitos dados que pertenciam a classe 2 como pertencentes a outra classe. Isso fica claro na matriz de confusão.

**4: UM PROBLEMA NÃO IMPLEMENTADO**

Desbalanceamento das classes e porque é mais vantajoso mantê-las assim.

A matriz de confusão mostra que um terço dos dados da segunda classe foi para atribuído erroneamente a uma outra classe que não a 2. Isto, juntamente com o número de amostras para cada classe (soma de cada linha da matriz de confusão) indica um desbalanceamento das classes. Podemos verificar isso pela função ***value_counts()*** da biblioteca ***pandas***:
```Python
df['fetal_health'].value_counts()
```

Como podemos ver, há uma quantidade bem maior de dados na classe 1 do que nas classes 2 e 3.

Existem duas formas de resolvermos esse desbalanceamento:


1.   **Oversampling**: replicamos, aleatoriamente, os dados das classes minoritárias até igualarmos a quantidade de dados da classe majoritária;
2.   **Undersampling**: descartamos, aleatoriamente, os dados das classes majoritárias até igualarmos a quantidade de dados da classe minoritária.

Porém, como a classe majoritária 1 é muito maior que a classe minoritária 3, os dados seriam de 3 seriam replicados quase dez vezes para compensar a diferença para a classe 1. Isso poderia acarretar em *overfitting*, fazendo com que nosso modelo se adaptasse muito bem ao processo de treinamento, mas respondesse muito mal à possíveis novas entradas. Por conta disso, o processo de *oversampling* foi descartado.

> Confira [aqui](https://github.com/gabrielestacio/projetoLAIS_fase2_ia/blob/master/C%C3%B3digos/C%C3%B3digos%20de%20testes/regressao_oversampling.py) o código de oversampling

O mesmo princípio se aplica ao *undersampling*: como a classe minoritária 3 é muito menor que a classe majoritária 1, descartaríamos muitos dados e teríamos conjuntos de treinamento e testes muito pequenos, com menos de 200 dados, o que prejudicaria profundamente a acurácia do modelo (perda de, aproximadamente, 12%) do modelo. Por isso, o *undersampling* também foi descartado.

> Confira [aqui](https://github.com/gabrielestacio/projetoLAIS_fase2_ia/blob/master/C%C3%B3digos/C%C3%B3digos%20de%20testes/regressao_undersampling.py) o código de undersampling

Sendo assim, aqui completamos o nosso projeto e atingimos a resolução do problema com um modelo de aprendizado de máquina preditivo multiclasse e com uma boa avaliação.
