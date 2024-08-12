import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# classes
#drop features
class Drop(BaseEstimator, TransformerMixin):
    def __init__(self, atr_drop = ['ano', 'id_aluno', 'turma']):
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
    def __init__(self, min_max_scaler = ['idade_aluno','ian','iaa',
                                         'ips','ipp','inde']):
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
            print('problema no one_hot')
            return dados

#ordinalfeature
class ordinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature = ['fase']):
        self.ordinal_feature = ordinal_feature
        
    def fit(self, dados):
        return self
    def transform(self, dados):
        if 'fase' in dados.columns:
            encoder = OrdinalEncoder()
            dados[self.ordinal_feature] = encoder.fit_transform(dados[self.ordinal_feature])
            return dados
        else:
            print('Fase não está no dataframe')
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