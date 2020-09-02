#Bibliotecas a serem utilizadas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

#Leitura dos dados a serem utilizados
df = pd.read_csv("C:/Users\gabri\OneDrive\Preparacao_Entrevista_Quarta\Machine_Learning\Wine_Project\wine_dataset.csv")
df['style'] = df['style'].replace(('red','white'),(0,1))
#Variável de predição [x] e alvo[y] - separando/organizando os dados
y = df['style']
x = df.drop('style',axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, train_size = 0.7)

#Treinando os dados com algoritmo de ML
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

#Valor de acurácia: dos n modelos de teste apresentados, quanto o algoritmo acertou em relação ao gabarito (y_teste)
resultado = modelo.score(x_teste, y_teste)
print(f'Acurácia: {resultado}')

#Verificando uma parte dos dados na predição
y1 = y_teste[600:603]
x1 = x_teste[600:603]
print(f'Gabarito: {y1}')
print(f'Dados dos vinhos: {x1}')
previsoes = modelo.predict(x1)
print(f'Previsão: {previsoes}')