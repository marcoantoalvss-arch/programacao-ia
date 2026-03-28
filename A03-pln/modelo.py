#import das bibliotecas
import pandas as pd #manipulação de dados em forma de tabela
import spacy #lib de processamento de linguagem natural
import joblib #salvar e carregar modelos de ia treindos
from sklearn.feature_extraction.text import TfidfVectorizer #converte texto em vetores
from sklearn.naive_bayes import MultinomialNB #classifica texto com base em probabilidade
from sklearn.pipeline import make_pipeline #junta várias etepas num fluxo só
from sklearn.model_selection import train_test_split #divide o conjunto de dados mem treino e teste
from sklearn.metrics import classification_report #avalia modelo

#ETAPA 1:carregar dados
print ("carregar dataset...")
df = pd.read_csv("dataset_chamados.csv")

#ETAPA 2: pipeline de procssamento focado em performace 
#vamos usar o Spacy dentro do fluxo da UI
nlp = spacy.load("pt_core_news_sm") #carregamento da lib da spacy em português

def prep(texto):
    doc = nlp(texto) #processamento do texto (tokenização e analise probabilistica)
    
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_punct #remove qualquer tipo de pontuação
               
    ])
print("Processando textos, podemos levar alguns instantes.")

df['texto_limpo']=df['texto'].apply(prep) #aplicar a função de limpeza na col do texto

#etapa 3: dividir entre treino e test
#X = texto de entrada
#y = categorias (labels)
X_train, X_test, y_train, y_test = train_test_split(
    df['texto_limpo'], #dados de entrada com pré processamento
    df['categoria'], #categorias
    test_size=0.2 #20 pra teste
    
)

#etapa 4: criar e treinar pipeline de ML
model_pipeline = make_pipeline(
    TfidfVectorizer(), #converter texto em valor númerico
    MultinomialNB()   #aplica clasificador Naive bayes (palavra : intenção/categoria)
)

#treinar modelo com os dados de treino
model_pipeline.fit(X_train, y_train)

#etapa 5 : salvar o modelo treinado
joblib.dump(model_pipeline, "modelo_triagem_suporte.pkl")
print("Modelo treinado e salvo como modelo_triagem_suporte.pkl")