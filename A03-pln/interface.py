import streamlit as st
import joblib
import spacy
import pandas as pd

#configuração da página (titulo eicone)
st.set_page_config(page_title="Triagem de chamados", page_icon="")

#carregamento de recursos -> em cache para que não seja necessário
#recarregar a cada clique
@st.cache_resource
def carregar_modelo():
   return joblib.load("modelo_triagem_suporte.pkl") #carregamento do modelo de ML treinado
    
@st.cache_resource
def carregar_nlp():
    nlp_model = spacy.load("pt_core_news_sm") #carregamento do modelo ML para lingua portuguesa
    if nlp_model is None:
        raise ValueError("Erro ao carregar o modelo")
    return nlp_model 

try:
    modelo = carregar_modelo()    
    nlp  = carregar_nlp()
    
except:
    st.error("Erro: Excute o script 'treinar_modelo_py' para gerar o arquivo .pkl")
    st.stop()
    
#lógica de processamento
def analisar_chamado(texto_usuario):
    
    #1.processamento linguistico com spacy
    doc = nlp(texto_usuario) 
    
    #extração de entidades nomeadas (ex.: AWS, locais, equipamentos e etc.)
    entidades = [(ent.text,ent.label_) for ent in doc.ents]
    
    #limpeza de texto pro modelo: lematizar, converter pra minusculo e remover pontuação 
    texto_limpo = " " .join({
        token.lemma_.lower()
        for token in doc 
        if not token.is_punct
    })   
    
    #2. predição com o modelo de machine learning
    
    #prevê a categoria do chamado
    categoria_predita = modelo.predict([texto_limpo])[0]
    #probabilidade de cada catgoria / classe
    probs = modelo.predict_proba([texto_limpo])[0]
    
    #pega maior probabilidade com resultado da analise
    confianca = max(probs)*100
    
    #retorna os resultados
    return categoria_predita, confianca, entidades
    #[infra - 80%, acesso - 65%, software - 30%]
    
#------interface gráfica------
st.title("Triagem de suporte") #titulo da página
st.markdown("Descreva o problema em poucas palavras.") #descrição

#criando o chat 
if "messages" not in st.session_state:
    st.session_state.messages = []
    
#exibir mensagens anteriores no chat
for message in st.session_state.messages: #cria um armazenamento pra cada mensagem
    with st.chat_message(message["role"]): #criar um botão de chat visual
        st.markdown(message["content"])  #exibir o texto da mensagem dentro do balão
        
#exibir o campo de entrada para o usuário (chat input)
if prompt:= st.chat_input("Ex.: O servidor AWS parou de responder..."):
    
    st.chat_message('user').markdown(prompt) 
    
    st.session_state.messages.append({
        "role":"user",
        "content": prompt   
    })    
          
#processar a resposta que a IA (modelo de ML) vai retornar ao usuário
    categoria, confianca, entidades = analisar_chamado(prompt)
 
#montar/personalizar a resposta em um formato amigável 
    resposta_md = f"""
**Análise do chamado:**
**Categoria:** `{categoria}`
**Confiança:** `{confianca:.2f}%`
"""  
        

    if entidades:    
        resposta_md +="\n\n **Entidades detectadas**"
        for ent in entidades:
                resposta_md += f"\n-*{ent[0]}*({ent[1]})"

#ações automáticas por categoria            
    acoes = {
    "Infraestrutura": "Encaminhando para euipe N2",
    "Acesso": "verificando logs de autenticação.",
    "Hardware": "Abrindo ordem de serviço.",
    "Software": "Verificando disponibilidade de licenças."
    }     
        #adicionar as ações sugestividade com base categoria
    resposta_md += f"\n\n **Ação** {acoes.get(categoria,'Triagem manual necessária.')}"

        #3. exibir a resposta do assistente

    with st.chat_message('assistant'):
        st.markdown(resposta_md)
            
        #salvar resposta no histórico
    st.session_state.messages.append({
        "role":"assistant",
        "content": resposta_md
    })    
                    