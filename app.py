import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import Drop, OneHot, minMax
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
# intro
st.image('images/img1.png')
st.markdown('Texto sobre a instituição')







#------------------------------------------------------------------------------------------
#TABS
tab1, tab2, tab3 = st.tabs(['Dashboard','Simulador de Ponto de Virada', 'Info'])

with tab1:
    st.subheader('Dashboard', divider = 'orange')
    st.image('images/STREAMLIT_-_Relatório_Datathon_Passos_Mágicos_page-0001.jpg')
    
    st.link_button('Link de Acesso ao Dashboard', 'https://lookerstudio.google.com/reporting/478820f6-2455-41d1-a56c-f887dd3dcddc')
    
with tab2:
    st.write('## Simulador de Ponto de Virada')
    
    st.markdown(''' "Passar pelo Ponto de Virada deve significar estar apto a iniciar a transformação da sua vida por meio da educação. Portanto, não se trata de um ponto de chegada, mas um momento no qual se inicia uma importante mudança. Ele deve ser o resultado a ser buscado pelas ações da Associação, em especial nas suas atividades educacionais e de socialização. A experiência de aprendizado e de convivência na Associação Passos Mágicos deve assim oferecer as condições e as oportunidades para que cada um de seus alunos, da sua forma e no seu tempo, desenvolvam as condições que os permitam passar pelo seu Ponto de Virada."  PDE2020: Roteiro de Avaliaçào do Indicador de Ponto de Virada (IPV)''')
    

    #------------------------------------------------------------------------------------------
    # Formulário
    #idade_aluno
    input_idade = int(st.slider('Selecione a idade do aluno', 7, 20))
    #fase
    input_fase = int(st.slider('Selecione a fase em que o aluno está', 0, 8))
    #pedra
    input_pedra = str(st.selectbox('Selecione a pedra do aluno:',
                                ('Ametista', 'Quartzo', 'Topázio', 'Ágata')))
    #ian
    input_ian = float(st.slider('Selecione o Indicador de adequação de nível do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #ida
    input_ida = float(st.slider('Selecione o Indicador de desempenho acadêmico do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #ieg
    input_ieg = float(st.slider('Selecione o Indicador de engajamento do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #iaa
    input_iaa = float(st.slider('Selecione o Indicador de Autoavaliação do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #ips
    input_ips = float(st.slider('Selecione o Indicador Psicossocial do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #ipp
    input_ipp = float(st.slider('Selecione o Indicador Psicopedagógico do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #ipv
    input_ipv = float(st.slider('Selecione o Indicador de Ponto de Virada do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #inde
    input_inde = float(st.slider('Selecione o Indice de Desenvolvimento Educacional do aluno:', 
                                    step=0.001, min_value = 0.0, max_value = 10.0))
    #instituição de ensino
    input_instituicao_ensino = str(st.selectbox('Selecione a Instituição de ensino do Aluno', ['Escola Pública', 'Escola Particular']))
    escola_dict = {'Escola Pública': 0, 'Escola Particular': 1}
    input_instituicao_ensino = escola_dict.get(input_instituicao_ensino)
    # anos na instituição
    input_anos_pm_2020 = int(st.slider('Selecione há quantos anos o aluno estuda na Passos Mágicos', 0, 5))

    #------------------------------------------------------------------------------------------

    # análise do aluno:
    # adicionei features aleatórias nas colunas que vão ser dropadas na pipeline para o modelo rodar tranquilamente

    novo_aluno = [input_idade, 
                input_fase,
                0, # ponto_de_virada 
                input_pedra,
                input_ian, 
                input_ida, 
                input_ieg, 
                input_iaa, 
                input_ips, 
                input_ipp, 
                input_ipv, 
                input_inde,
                input_instituicao_ensino,
                input_anos_pm_2020
                ]

    #------------------------------------------------------------------------------------------
        
    # separando dados de treino e teste
    from sklearn.model_selection import train_test_split
    SEED = np.random.seed(42)
    dados_treino, dados_teste = train_test_split(dados, test_size=0.2, random_state=SEED)

    #criando df de novo aluno
    novo_aluno_df = pd.DataFrame([novo_aluno], columns=dados.columns)

    #concatenando
    dados_e_novo_aluno = pd.concat([dados_teste, novo_aluno_df], ignore_index=True)
    
    dados_e_novo_aluno.to_csv('novo_aluno.csv', index=False)

    #------------------------------------------------------------------------------------------
    #Pipeline
    def pipeline(dados):
        pipeline = Pipeline([
            ('DropFeatures', Drop()),
            ('OneHotEncoder', OneHot()),
            ('MinMaxScaler', minMax()),
        ])
        dados_pipeline = pipeline.fit_transform(dados)
        return dados_pipeline

    # aplicando a pipeline
    dados_e_novo_aluno = pipeline(dados_e_novo_aluno)

    # retirando o target
    pv_predito = dados_e_novo_aluno.drop(['ponto_de_virada'], axis=1)
    

    #------------------------------------------------------------------------------------------
    if st.button('enviar'):
        modelo = joblib.load('modelo/logistic_regression.joblib')
        predicao = modelo.predict(pv_predito)
        if predicao[-1] == 1:
            st.write('### Este aluno está apto a atingir o Ponto de Virada')
        else:
            st.error('### Este aluno vai ter dificuldades em atingir o Ponto de Virada')

with tab3:
    st.subheader('Informações sobre o trabalho:', divider = 'orange')
    st.markdown('**GRUPO 46: Giuliana de Sousa Fernandes RM352002**')


    st.link_button('GitHub do projeto', 'https://github.com/giulianafernandes/datathon_fiap')