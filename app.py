import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import Drop, minMax, oversample, OneHot, ordinalFeature
import joblib
from joblib import load
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

#------------------------------------------------------------------------------------------

st.header(':red[DATATHON:] Passos Mágicos', divider = 'rainbow')

#carregando o df
dados = pd.read_csv('dados/csv_tratado/dados_ML.csv')

#------------------------------------------------------------------------------------------

st.write('## Simulador de Ponto de Virada')

#escrever algum texto aqui

#------------------------------------------------------------------------------------------

#idade_aluno
input_idade = int(st.slider('Selecione a idade do aluno', 7, 20))
#fase
input_fase = int(st.slider('Selecione a fase em que o aluno está', 0, 8))
#pedra
input_pedra = str(st.selectbox('Selecione a pedra do Aluno:',
                            ('Ametista', 'Quartzo', 'Topázio', 'Ágata')))
#ian
input_ian = float(st.slider('Selecione o Ian do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#ida
input_ida = float(st.slider('Selecione o Ida do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#ieg
input_ieg = float(st.slider('Selecione o Ieg do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#iaa
input_iaa = float(st.slider('Selecione o Iaa do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#ips
input_ips = float(st.slider('Selecione o Ips do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#ipp
input_ipp = float(st.slider('Selecione o Ipp do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#ipv
input_ipv = float(st.slider('Selecione o Ipv do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))
#inde
input_inde = float(st.slider('Selecione o Inde do aluno e pressione enter', 
                                step=0.001, min_value = 0.0, max_value = 10.0))

#------------------------------------------------------------------------------------------

# análise do aluno:
# adicionei features aleatórias nas colunas que vão ser dropadas na pipeline para o modelo rodar tranquilamente
#'ida', 'ieg', 'fase', 'ipv'
# AJUSTAR O DROP 'idade_aluno','pedra'

novo_aluno = ['pm-12667', #'id_aluno'
            input_idade,
            input_fase,
            'P', #'turma' 
            0, #target - ponto de virada
            input_pedra,
            input_ian,
            #input_ida,
            #input_ieg,
            input_iaa,
            input_ips,
            input_ipp,
            #input_ipv,
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
# novo_aluno_df.to_csv('novo_aluno.csv', index=False)

#concatenando
dados_e_novo_aluno = pd.concat([dados_teste, novo_aluno_df], ignore_index=True)

#------------------------------------------------------------------------------------------
#Pipeline
def pipeline(dados):
    pipeline = Pipeline([
        ('DropFeatures', Drop()),
        ('MinMaxScaler', minMax()),
        ('OneHotEncoder', OneHot()),
        ('OrdinalFeature', ordinalFeature()),
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
    modelo = joblib.load('modelo/gradient_boosting.joblib')
    predicao = modelo.predict(pv_predito)
    if predicao[-1] == 1:
        st.write('### Este aluno está apto a atingir o Ponto de Virada')
    else:
        st.error('### Este aluno vai ter dificuldades em atingir o Ponto de Virada')

#pedra?????


#with tab4:
#st.subheader('Informações sobre o trabalho:', divider = 'orange')
#st.markdown('**GRUPO ?: Giuliana de Sousa Fernandes RM352002**')


#st.link_button('GitHub do projeto', 'https://github.com/giulianafernandes/datathon_fiap')'''