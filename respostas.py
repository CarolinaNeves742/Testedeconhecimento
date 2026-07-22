import pandas as pd
import json 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

'''
# 1. Carregar dados de exemplo (Classificação de tumor)
dados = load_breast_cancer()
X, y = dados.data, dados.target

# 2. Dividir os dados em treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicializar e treinar o modelo de Regressão Logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# 4. Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_test)

# 5. Avaliar a acurácia do modelo
precisao = accuracy_score(y_test, previsoes)
print(f'Acurácia do modelo: {precisao:.2f}')
'''
# Dados

with open('./Dados/Dados.jsonl', 'r') as arquivo:
    linhas = arquivo.readlines()
    print(f'Número de linhas no arquivo: {len(linhas)}')

data = []
target = []
palavras_positivas_ingles = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'wonderful', 'best', 'awesome', 'perfect']
palavras_negativas_ingles = ['bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'disappointing', 'poor', 'mediocre', 'unacceptable','Defective']
for i, linha in enumerate(linhas[:200]):
    linha_json = json.loads(linha)
    if linha_json['rating'] >= 4 :
        rat = 1
        count = sum(1 for palavra in palavras_positivas_ingles if palavra in linha_json['text'])
        if count >= 3:
            rat_txt = 1
        elif linha_json['rating'] < 3 and linha_json['rating'] >= 1:
            rat_txt = 0.5
        else:
            rat_txt = 0 
    else:
        rat = 0
        count = sum(1 for palavra in palavras_negativas_ingles if palavra in linha_json['text'])
        if count >= 3:
            rat_txt = -1
        elif linha_json['rating'] < 3 and linha_json['rating'] >= 1:
            rat_txt = -0.5
        else:
            rat_txt = 0
    temp=[rat_txt]
    data.append(temp)
    target.append(rat)
# Iniciando análise dos dados
X, y =  data, target
# 2. Dividir os dados em treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicializar e treinar o modelo de Regressão Logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# 4. Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_test)

# 5. Avaliar a acurácia do modelo
precisao = accuracy_score(y_test, previsoes)
print(f'Acurácia do modelo: {precisao:.2f}')

    

