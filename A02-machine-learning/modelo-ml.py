#etapa 1 importando todos os módulos necessarios 
import pandas as pd #ferramentas para criar e alterar dados em tabelas
import numpy as np #ferramentas dr análise matemática

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns 
import matplotlib.pyplot as pyplot
import joblib

#etapa 2: importar dicionário com dados "dataset"

try:
    print("carregando arquivo 'churn-data.csv...")
    df = pd.read_csv('churn-data.csv') #ler o arquivo e criar uma tabela
    print("sucesso, {len(df)} linhas importadas")
    
except FileNotFoundError:
    print("O arquivo não foi encontrado na pasta")
    exit()
    
#etapa 3 : pre processamento de dados (preparar a IA para ser treinada)
#passo 1 : separar a pergunta (x) da resposta (y)
# (x) -> tudo menos a coluna cancelou< são as "pista" pro modelo
X=df.drop('cancelou', axis=1)
#(y) -> Apenas a coluna 'cancelou', é o que queremos que o modelo preveja
y = df['cancelou']

#passo 2: dividir o treino do teste
X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size= 0.2, random_state= 42)
#teste_size=0.2 separa 20% da massa de dados para testar o modelo

#passo 3 : normalizando (colocar tudo na mesma escala)
scaler = StandardScaler()

#fit transform do treino: IA  calcula a média e desvio de padrão de treino 
X_train_scaled = scaler.fit_transform(X_train)
#transform no teste: usamos a régua calculada no treino
X_test_sclaed = scaler.transform(X_test)

#etapa 4: treinar o modelo e realizar a previsão de dados 
#criando o modelo
#n_estimators=100 , cria 100 arvores de decisão 
modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42)

#treinar/ajustar a IA
modelo_churn.fit(X_train_scaled, y_train)

#prever as respostas
previsoes = modelo_churn.predict(X_test_sclaed)


#etapa 5: avaliação do nosso módelo 
print("relatorio de performance")
print(classification_report(y_test,previsoes))

 #etapa 6 : Deploy -> salvar o trabalho
joblib.dump(modelo_churn,'modelo_churn_v1.pkl')

joblib.dump(scaler, 'padronizador_v1.pk1')
print("Arquivos de ML foram exportados com sucesso")
