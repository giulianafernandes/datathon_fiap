import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# classes


#drop features
class Drop(BaseEstimator, TransformerMixin):
    def __init__(self, atr_drop = ['ano', 'id_aluno', 'turma', 'pedra']):
        self.atr_drop = atr_drop
    def fit(self, dados):
        return self
    def transform(self, dados):
        if (set(self.atr_drop).issubset(dados.columns)):
            dados.drop(self.atr_drop, axis=1, inplace=True)
            return dados
        else:
            print('Uma ou mais features não estão no DataFrame')
            return dados
        
#min_max
class minMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler = ['idade_aluno', 'fase','ian', 'ida', 'ieg', 
                                         'iaa', 'ips', 'ipp', 'ipv', 'inde']):
        self.min_max_scaler = min_max_scaler
    def fit(self, dados):
            return self
    def transform(self, dados):
        if (set(self.min_max_scaler).issubset(dados.columns)):
            min_max = MinMaxScaler()
            dados[self.min_max_scaler] = min_max.fit_transform(dados[self.min_max_scaler])
            return dados
        else:
            print('Uma ou mais features não estão no DataFrame')
            return dados
        
#oversample       
class oversample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, dados):
        return self
    def transform(self, dados):
        if 'ponto_de_virada' in dados.columns:
            over_sample = SMOTE(sampling_strategy='minority')
            X_balanced, y_balanced = over_sample.fit_resample(dados.loc[:, dados.columns != 'ponto_de_virada'], 
                                                              dados['ponto_de_virada'])
            dados_balanceados = pd.concat([pd.DataFrame(X_balanced), pd.DataFrame(y_balanced)], axis=1)
            return dados_balanceados
        else:
            print('O target não está no DataFrame')
            return dados