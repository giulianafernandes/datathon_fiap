import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# classes
#drop features
class Drop(BaseEstimator, TransformerMixin):
    def __init__(self, atr_drop = ['fase']):
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
        
#onehot
class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self, onehotenc = ['pedra']):
        self.onehotenc = onehotenc
    def fit(self, dados):
        return self
    def transform(self, dados):
        if set(self.onehotenc).issubset(dados.columns):
            def encoder(dados, onehotenc):
                encoder = OneHotEncoder()
                encoder.fit(dados[onehotenc])
                
                feature_names = encoder.get_feature_names_out(onehotenc)
                dados = pd.DataFrame(encoder.transform(dados[self.onehotenc]).toarray(),
                                     columns = feature_names, index = dados.index)
                return dados
            def concatenando(dados, dados_enc, onehotenc):
                another_features = [feature for feature in dados.columns if feature not in onehotenc]
                dados = pd.concat([dados_enc, dados[another_features]], axis=1)
                return dados
            
            dados_encoded = encoder(dados, self.onehotenc)
            
            dados = concatenando(dados, dados_encoded, self.onehotenc)
            return dados
        
        else:
            print('Uma ou mais features não estão no Dataframe')
            return dados
        
# minmax

class minMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler = ['idade_aluno','ian', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 
                                         'ipv', 'inde', 'anos_pm_2020']):
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