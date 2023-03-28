"""Criando modelo de Machine Learning utilizando-se o Iris Data Set. Onde 'CLASS' é a saída a ser prevista.
   Esse Data Set contém os seguintes atributos:
   1. sepal length in cm (SL)
   2. sepal width in cm (SW)
   3. petal length in cm (PL)
   4. petal width in cm (PW)
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica (CLASS)"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle

url = 'iris.data'
name = ['SL', 'SW', 'PL', 'PW', 'CLASS']
iris = pd.read_csv(url, names=name)

# Análise das variáveis
print(f'Database Iris início:\n{iris.head()}\nDatabase Iris fim:\n{iris.tail()}\nDescrição das váriáveis:\n'
      f'{iris.describe()}\nCorrelação entre as variáveis\n{iris.corr()}')

# Gráficos
scatter_matrix(iris)
iris.hist()
plt.show()

# Padronizar dados
array = iris.values
X = array[:, 0:4]
Y = array[:, 4]
scaler = StandardScaler()
rescaledX = scaler.fit_transform(X)
print(f'Dados padronizados: \n{rescaledX[0:5, :]}')

# K fold cross validation
kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# Testando qual é o algoritmo mais preciso
models = [('LR', LogisticRegression(solver='liblinear')), ('LDA', LinearDiscriminantAnalysis()),
          ('RF', RandomForestClassifier()), ('DecTree', DecisionTreeClassifier())]

for name, model in models:
    results = cross_val_score(model, rescaledX, Y, cv=kfold)
    print(f"\nKfold com {name}- Precisão: {results.mean() * 100.0:.3f}% "
          f",Desvio Padrão: ({results.std() * 100.0:.3f}%)")

# Escolhendo melhor configuração do parâmetro 'solver' do LDA com GridSearchCV
param_grid = dict()
param_grid['solver'] = ['svd', 'lsqr', 'eigen']
model = LinearDiscriminantAnalysis()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold)
grid.fit(rescaledX, Y)
print(f'\nPrecisão do algoritmo com GridSearchCV: {grid.best_score_}, melhor solver: {grid.best_estimator_.solver}\n')

# Modelo final com toda a Database
modelo_final = LinearDiscriminantAnalysis(solver='svd')
modelo_final.fit(rescaledX, Y)

# Salvando o modelo
filename = 'modelo.sav'
pickle.dump(modelo_final, open(filename, 'wb'))

# Exemplo caso haja necessidade de testar o modelo
loaded_model = pickle.load(open(filename, 'rb'))
# Os dados inseridos em score são ilustrativos, o mais preciso é inserir dados não usados no fit do modelo
result = loaded_model.score(rescaledX[0:5, :], Y[0:5])
print(f'Resultado do modelo testando novas entradas: {result * 100.0:.3f}%')
