#importar modulos
import streamlit as st #lib que transforma em python em site
import joblib #salva e exporta o modelo de IA
import numpy as np #lib para organizar os dados numericos

#passo 1: configurando a aba do navegador
st.set_page_config(page_title="análise de churn",page_icon="")

#textos tela principal
st.title("sistema de retenção de base") #titulo da pagina
st.markdown("insira dados do cliente para verificar risco de cancelamento ")

#passo 2: importar os dados da inteligencia artificial com o joblib
modelo = joblib.load('modelo_churn_v1.pkl') #Carregando as informação do modelo, regras de decisão o modelo
scaler = joblib.load('padronizador_v1.pk1')#carregando a régua matematica do modelo

#passo 03: interface de entrada com um formulario
col1, col2 = st.columns(2)#quantidade de colunas que vc vai criar 


#coluna lado esquerdo(col1)
with col1:
    tempo = st.number_input("tempo de contrato (meses)", min_value=1, value=12, max_value=200)#number_input -> cria a entrada de numeros
    valor = st.number_input("valor da assinatura: (R$)", min_value=0.0, value=50.0)
with col2:
     reclamacoes = st.slider("Histórico de reclamações", 0,10,1)

#passo 4 :  processamento de dados
if st.button("Analisar risco"):
    dados = scaler.transform([[tempo, valor, reclamacoes]]) 
    probabilidade = modelo.predict_proba(dados)[0][1]#previsão de probabilidade

#passo 5: feedback de negocios
    st.divider() #cria linha divisoria

#probabilidade <70%
    if probabilidade >0.7:
     st.error(f"*ALTO RISCO DE CHURN*({probabilidade*100:.1f}%)")
     st.info("Sugestão de ação:* Oferecer cupom de fidelidade: FID210330OFF")
     
    elif probabilidade >0.3:
        st.warning(F"*Risco moderado de churn*  ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:* Realizar chamada de acompanhamento.")
    
    else:
        st.success(F"*Cliente estável ({probabilidade*100:.1f}%)")
        st.info("Nada a realizar no momento")    
    
                                                                