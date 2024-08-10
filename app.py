import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import Drop, minMax, oversample
import joblib
from joblib import load
import numpy as np

#------------------------------------------------------------------------------------------

#carregando o df
dados = pd.read_csv('dados/csv_tratado/dados_ML.csv')

#------------------------------------------------------------------------------------------

st.write('## Simulador de Ponto de Virada')

#escrever algum texto aqui

#------------------------------------------------------------------------------------------

#idade_aluno
st.write('Idade do Aluno')
input_idade = int(st.slider('Selecione a idade do aluno', 7, 20))
#fase
st.write('Fase')
input_fase = int(st.slider('Selecione a fase em que o aluno está', 0, 8))
#ian
st.write('Ian')
input_ian = float(st.text_input('Digite o Ian do aluno e pressione enter', 0))
#ida
st.write('Ida')
input_ida = float(st.text_input('Digite o Ida do aluno e pressione enter', 0))
#ieg
st.write('Ieg')
input_ieg = float(st.text_input('Digite o Ieg do aluno e pressione enter', 0))
#iaa
st.write('Iaa')
input_iaa = float(st.text_input('Digite o Iaa do aluno e pressione enter', 0))
#ips
st.write('Ips')
input_ips = float(st.text_input('Digite o Ips do aluno e pressione enter', 0))
#ipp
st.write('Ipp')
input_ipp = float(st.text_input('Digite o Ipp do aluno e pressione enter', 0))
#ipv
st.write('Ipv')
input_ipv = float(st.text_input('Digite o Ipv do aluno e pressione enter', 0))
#inde
st.write('Inde')
input_inde = float(st.text_input('Digite o Inde do aluno e pressione enter', 0))

#------------------------------------------------------------------------------------------

# análise do aluno:
# adicionei features aleatórias nas colunas que vão ser dropadas para o modelo rodar tranquilamente

novo_aluno = ['pm-12667', #'id_aluno'
              input_idade,
              input_fase,
              'P', #'turma' 
              0, #target - ponto de virada
              'Ametista', #'pedra'
              input_ian,
              input_ida,
              input_ieg,
              input_iaa,
              input_ips,
              input_ipp,
              input_ipv,
              input_inde,
              '2021-01-01', #'ano'
              ]

#------------------------------------------------------------------------------------------
# separando dados de treino e teste
def split_dados(df, test_size):
    SEED = np.random.seed(42)
    dados_treino, dados_teste = train_test_split(df, test_size=test_size, random_state=SEED)
    return dados_treino.reset_index(drop=True), dados_teste.reset_index(drop=True)

dados_treino, dados_teste = split_dados(dados, 0.2)

#criando df de novo aluno
novo_aluno_df = pd.DataFrame([novo_aluno], columns=dados.columns)

#concatenando
dados_e_novo_aluno = pd.concat([dados_teste, novo_aluno_df], ignore_index=True)

#------------------------------------------------------------------------------------------
#Pipeline
def pipeline(dados):
    pipeline = Pipeline([
        ('DropFeatures', Drop()),
        ('MinMaxScaler', minMax()),
        ('SMOTE', oversample())
    ])
    dados = pipeline.fit_transform(dados)
    return dados

# aplicando a pipeline
dados_e_novo_aluno = pipeline(dados_e_novo_aluno)

# retirando o target
pv_predito = dados_e_novo_aluno.drop(['ponto_de_virada'], axis=1)

#------------------------------------------------------------------------------------------
if st.button('enviar'):
    modelo = joblib.load('modelo/logistic_regression.joblib')
    predicao = modelo.predict(pv_predito)
    if predicao[-1] == 1:
        st.success('### Esse aluno esta apto a atingir o Ponto de Virada')
    else:
        st.error('### Esse aluno vai ter dificuldades em atingir o Ponto de Virada')

#pedra?????