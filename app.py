import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
import numpy as np


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

# análise do aluno:
novo_aluno = [input_idade,
              input_fase,
              input_ian,
              input_ida,
              input_ieg,
              input_iaa,
              input_ips,
              input_ipp,
              input_ipv,
              input_inde,
              0 #target
              ]


def split_dados(df, test_size):
    SEED = np.random.seed(42)
    dados_treino, dados_teste = train_test_split(df, test_size=test_size, random_state=SEED)
    return dados_treino.reset_index(drop=True), dados_teste.reset_index(drop=True)


novo_aluno_df = pd.DataFrame([novo_aluno], columns=dados.columns) ############ verificar isso








#pedra?????